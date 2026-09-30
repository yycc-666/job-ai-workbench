"""检索模块的单元测试。运行：pytest -q"""
import pytest

from search import JobSearcher, tokenize, extract_job_id


@pytest.fixture(scope="session")
def searcher():
    """整个测试会话只加载一次模型和数据。"""
    return JobSearcher()


def test_jd_loaded(searcher):
    """数据必须真的加载进来了。"""
    assert len(searcher.jd_meta) > 0
    assert len(searcher.chunk_vectors) > 0


def test_search_respects_top_k(searcher):
    """top_k 是上限，不是承诺——返回条数不能超过它。"""
    for k in (1, 5, 10):
        assert len(searcher.search("开发", top_k=k)["results"]) <= k


def test_same_query_same_result(searcher):
    """同一个查询两次调用，顺序必须一致。

    这一条防的是"结果不可复现"——比如集合遍历顺序、
    浮点排序不稳定之类的问题。评测要能重跑，这是前提。
    """
    a = searcher.search("武汉 Python 开发", top_k=10)["results"]
    b = searcher.search("武汉 Python 开发", top_k=10)["results"]
    assert [x["job_id"] for x in a] == [x["job_id"] for x in b]


def test_city_filter_applies(searcher):
    """识别到城市约束时，返回的岗位必须全部属于该城市。"""
    r = searcher.search("武汉的后端开发岗位", top_k=10)
    assert r["city_constraint"] == "武汉市"
    assert r["results"], "城市过滤后不应该返回空"
    assert all(x["城市"] == "武汉市" for x in r["results"])


def test_result_fields_complete(searcher):
    """返回字段是接口契约，缺字段会让前端炸掉。"""
    r = searcher.search("测试", top_k=3)
    for item in r["results"]:
        for field in ("job_id", "岗位名", "公司", "城市", "url", "score"):
            assert field in item, f"缺少字段 {field}"


def test_tokenize_drops_stopwords():
    tokens = tokenize("熟悉 Python，有相关经验，能承受压力")
    assert "python" in tokens
    assert "熟悉" not in tokens          # 停用词表里有它
    assert "相关" not in tokens


def test_extract_job_id():
    assert extract_job_id("https://whpu.91wllm.cn/id/2384397") == "2384397"
    assert extract_job_id("") == ""
    assert extract_job_id(None) == ""