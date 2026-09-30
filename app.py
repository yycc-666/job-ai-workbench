# app.py
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
from pydantic import BaseModel, Field

from search import JobSearcher

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global searcher, redis_client
    print("启动：加载 JobSearcher...")
    searcher = JobSearcher()

    print("启动：连接 Redis...")
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=2,       # 连接阶段超时
            socket_timeout=2,               # 读写阶段超时
            retry=Retry(NoBackoff(), 0),     # 不重试：缓存超时直接交给上层降级
        )
        redis_client.ping()
        print("Redis 连接成功")
    except Exception as e:
        print(f"Redis 连接失败：{e}，将不使用缓存")
        redis_client = None

    print("启动完成")
    yield
    print("关闭")


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
        except Exception as e:
            print(f"缓存读取失败：{e}")

    # 未命中，检索
    result = searcher.search(req.query, top_k=req.top_k)
    result["cached"] = False

    if redis_client:
        try:
            redis_client.setex(key, CACHE_TTL, json.dumps(result, ensure_ascii=False))
        except Exception as e:
            print(f"缓存写入失败：{e}")

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
        return {"status": "cleared", "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清缓存失败：{e}")


@app.get("/metrics")
def metrics():
    """Prometheus 抓取端点"""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)