# app.py
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
from pydantic import BaseModel, Field

from search import JobSearcher

# ========== 日志 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("job-search")

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = 3600

searcher = None
redis_client = None


# ========== Prometheus 指标 ==========
MATCH_REQUESTS = Counter(
    "match_requests_total",
    "Total /match requests",
    ["cached"]  # 标签：true / false
)
MATCH_LATENCY = Histogram(
    "match_latency_seconds",
    "/match latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
INTERVIEW_REQUESTS = Counter(
    "interview_requests_total",
    "Total /interview requests",
    ["status"]  # 标签：found / not_found
)
CACHE_HITS = Counter(
    "cache_hits_total",
    "Total cache hits"
)
# 进程常驻内存：用来给 K8s 的 requests 提供依据。
# 这里报的是"当前 RSS"，适合定 requests；
# 模型加载那一刻的峰值会更高，那部分交给 limits 兜住。
MEMORY_BYTES = Gauge(
    "app_resident_memory_bytes",
    "Resident memory of the app process in bytes (Linux only)"
)


def read_rss_bytes():
    """读当前进程的常驻内存（字节）。非 Linux 环境（比如 Windows 本机）返回 None。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024   # /proc 里的单位是 KB
    except Exception:
        pass
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global searcher, redis_client
    logger.info("启动：加载 JobSearcher")
    searcher = JobSearcher()

    logger.info("启动：连接 Redis host=%s port=%s", REDIS_HOST, REDIS_PORT)
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=2,        # 连接阶段超时
            socket_timeout=2,                # 读写阶段超时
            retry=Retry(NoBackoff(), 0),     # 不重试：缓存超时直接交给上层降级
        )
        redis_client.ping()
        logger.info("Redis 连接成功")
    except Exception as e:
        # 缓存只是加速手段，连不上不影响核心检索，所以是 warning 不是 error
        logger.warning("Redis 连接失败，将不使用缓存：%s", e)
        redis_client = None

    logger.info("启动完成，共加载 %d 条 JD", len(searcher.jd_meta))
    yield
    logger.info("服务关闭")


app = FastAPI(title="求职 AI 工作台 API", lifespan=lifespan)


class MatchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(10, ge=1, le=50)


def make_cache_key(query: str, top_k: int) -> str:
    h = hashlib.md5(f"{query}|{top_k}".encode("utf-8")).hexdigest()
    return f"match:{h}"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "jd_count": len(searcher.jd_meta) if searcher else 0,
        "cache": "on" if redis_client else "off",
    }


@app.post("/match")
def match(req: MatchRequest):
    start = time.time()

    if searcher is None:
        MATCH_REQUESTS.labels(cached="false").inc()
        MATCH_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=503, detail="服务未就绪")

    key = make_cache_key(req.query, req.top_k)

    # 命中缓存
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                CACHE_HITS.inc()
                MATCH_REQUESTS.labels(cached="true").inc()
                MATCH_LATENCY.observe(time.time() - start)
                return data
        except Exception:
            # exception 会自动带上堆栈；不写 as e 是因为不需要再拼消息
            logger.exception("缓存读取失败")

    # 未命中，检索
    result = searcher.search(req.query, top_k=req.top_k)
    result["cached"] = False

    if redis_client:
        try:
            redis_client.setex(key, CACHE_TTL, json.dumps(result, ensure_ascii=False))
        except Exception:
            logger.exception("缓存写入失败")

    MATCH_REQUESTS.labels(cached="false").inc()
    MATCH_LATENCY.observe(time.time() - start)
    return result


@app.get("/interview/{job_id}")
def interview(job_id: str):
    if searcher is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    result = searcher.get_interview_questions(job_id)
    if not result:
        INTERVIEW_REQUESTS.labels(status="not_found").inc()
        raise HTTPException(status_code=404, detail=f"没有 {job_id} 的面试题")
    INTERVIEW_REQUESTS.labels(status="found").inc()
    return result


# 本服务允许清理的 key 前缀。写成常量而不是参数——
# 一旦前缀由调用方决定，传 ?prefix=* 就等于清空整库。
CACHE_PREFIXES = ("match:",)


@app.delete("/cache")
def clear_cache():
    """只清理本服务的缓存键（前缀 match:）。"""
    if not redis_client:
        return {"status": "no cache"}

    deleted = 0
    try:
        for prefix in CACHE_PREFIXES:
            batch = []
            for key in redis_client.scan_iter(match=f"{prefix}*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    redis_client.delete(*batch)      # 批量删，少很多网络往返
                    deleted += len(batch)
                    batch.clear()
            if batch:
                redis_client.delete(*batch)
                deleted += len(batch)
        logger.info("清缓存完成，删除 %d 个键", deleted)
        return {"status": "cleared", "deleted": deleted}
    except Exception as e:
        logger.exception("清缓存失败")
        raise HTTPException(status_code=500, detail=f"清缓存失败：{e}")


@app.get("/metrics")
def metrics():
    """Prometheus 抓取端点。顺便刷新一次内存指标。"""
    rss = read_rss_bytes()
    if rss is not None:
        MEMORY_BYTES.set(rss)
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
