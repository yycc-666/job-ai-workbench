"""求职 AI 工作台 · 简易界面（Streamlit）

它不直接读数据、不加载模型，只做一件事：调用你已经做好的 HTTP 接口。

用法（两个终端）：
  终端 1：uvicorn app:app --port 8001
  终端 2：streamlit run app_ui.py
"""
import requests
import streamlit as st

API = "http://127.0.0.1:8001"

st.set_page_config(page_title="求职 AI 工作台", page_icon="🔍", layout="wide")
st.title("求职 AI 工作台")
st.caption("输入一句查询，返回匹配岗位；点开任意岗位，看它可能问的面试题。")

# ── 后端是否在线 ──
try:
    health = requests.get(f"{API}/health", timeout=3).json()
except Exception:
    st.error("连不上后端。请先在另一个终端运行：`uvicorn app:app --port 8001`")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("岗位总数", health.get("jd_count", 0))
c2.metric("缓存", "已启用" if health.get("cache") == "on" else "未启用")
c3.metric("状态", health.get("status", "?"))

st.divider()

# ── 搜索区 ──
col1, col2 = st.columns([4, 1])
query = col1.text_input(
    "想找什么岗位？", "武汉 后端开发",
    help="可以带上城市、岗位方向、学历，例如「武汉 后端开发 本科」",
)
top_k = col2.number_input("返回条数", min_value=1, max_value=50, value=5)

if st.button("搜索", type="primary"):
    if not query.strip():
        st.warning("请输入查询内容")
    else:
        with st.spinner("检索中…"):
            try:
                r = requests.post(
                    f"{API}/match",
                    json={"query": query, "top_k": int(top_k)},
                    timeout=30,
                )
            except Exception as e:
                st.error(f"请求失败：{e}")
                st.stop()
        if r.status_code != 200:
            st.error(f"检索失败：{r.status_code} {r.text[:200]}")
            st.stop()

        data = r.json()
        tip = f"共 {len(data['results'])} 条"
        if data.get("city_constraint"):
            tip += f"　·　已按城市过滤：{data['city_constraint']}"
        if data.get("edu_constraint"):
            tip += f"　·　学历约束：{data['edu_constraint']}"
        st.caption(tip)

        for i, job in enumerate(data["results"], 1):
            with st.container(border=True):
                a, b = st.columns([4, 1])
                a.markdown(f"**{i}. {job['岗位名']}**")
                a.caption(f"{job['公司']} · {job['城市']}　|　融合分 {job['score']}")
                b.link_button("查看原 JD", job["url"])
                if b.button("看面试题", key=f"btn-{job['job_id']}"):
                    st.session_state["open_job"] = job["job_id"]

# ── 面试题区 ──
jid = st.session_state.get("open_job")
if jid:
    st.divider()
    st.subheader(f"岗位 {jid} · 预测面试题")
    try:
        qr = requests.get(f"{API}/interview/{jid}", timeout=10)
    except Exception as e:
        st.error(f"请求失败：{e}")
        st.stop()

    if qr.status_code == 404:
        st.info("这个岗位还没有生成面试题——题库目前只覆盖 30 个目标岗位。")
    else:
        for n, item in enumerate(qr.json()["questions"], 1):
            st.markdown(f"**{n}. [{item.get('category', '')}]** {item.get('question', '')}")
            if item.get("why"):
                st.caption(f"出题意图：{item['why']}")
