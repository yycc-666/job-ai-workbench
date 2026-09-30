"""接口层测试。用 FastAPI 的 TestClient，会触发 lifespan（加载模型，较慢）。"""
import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(scope="session")
def client():
    # 用 with 才能触发 lifespan，否则 searcher 是 None
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["jd_count"] > 0


def test_match_returns_results(client):
    r = client.post("/match", json={"query": "武汉 后端开发", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) <= 3
    assert "cached" in body


@pytest.mark.parametrize("payload", [
    {"query": "", "top_k": 5},              # 空查询
    {"query": "x", "top_k": 0},             # 下越界
    {"query": "x", "top_k": 999},           # 上越界
    {"query": "x" * 500, "top_k": 5},       # 超长
    {"top_k": 5},                           # 缺必填字段
])
def test_match_rejects_bad_request(client, payload):
    """参数校验必须拦住非法请求。这是接口的第一道防线。"""
    assert client.post("/match", json=payload).status_code == 422


def test_interview_not_found(client):
    r = client.get("/interview/99999999999")
    assert r.status_code == 404


def test_cache_hit_when_redis_available(client):
    """有 Redis 时，第二次相同请求应该命中缓存。"""
    if client.get("/health").json().get("cache") != "on":
        pytest.skip("当前没有 Redis，跳过缓存测试")
    payload = {"query": "缓存往返测试专用查询", "top_k": 3}
    first = client.post("/match", json=payload).json()
    second = client.post("/match", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True


def test_clear_cache_keeps_other_keys(client):
    """清缓存只能清自己的前缀，不能动别人的数据。"""
    if client.get("/health").json().get("cache") != "on":
        pytest.skip("当前没有 Redis，跳过")
    r = app_module.redis_client
    r.set("other-service:key", "不许被删")
    try:
        assert client.delete("/cache").status_code == 200
        assert r.get("other-service:key") == "不许被删"   # ← 关键断言
    finally:
        r.delete("other-service:key")


@pytest.mark.parametrize("evil_prefix", ["*", "", "other-service:"])
def test_clear_cache_ignores_prefix_param(client, evil_prefix):
    """接口不接受前缀参数——传了也必须无效。

    回归保护：如果以后有人把 prefix 参数加回来，
    传 ?prefix=* 就等于清空整库，这条测试会立刻失败。
    """
    if client.get("/health").json().get("cache") != "on":
        pytest.skip("当前没有 Redis，跳过")
    r = app_module.redis_client
    r.set("other-service:key", "不许被删")
    try:
        resp = client.delete("/cache", params={"prefix": evil_prefix})
        assert resp.status_code == 200
        assert r.get("other-service:key") == "不许被删"   # ← 关键断言
    finally:
        r.delete("other-service:key")