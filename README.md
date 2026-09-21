# 求职 AI 工作台

一个自己每天用的求职工具：输入查询，输出匹配岗位 + 面试题。

从零实现 RAG 检索链路——数据采集、LLM 结构化抽取、混合检索（向量 + BM25）、FastAPI 服务化、面试题生成。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![BGE](https://img.shields.io/badge/BGE--small--zh--v1.5-embedding-orange)

---

## 核心能力

| 能力 | 实现 | 关键数字 |
| --- | --- | --- |
| 数据采集 | DrissionPage + 合规白名单 | 292 条 JD |
| 结构化抽取 | LLM + Prompt 工程 | 准确率 **94%** |
| 混合检索 | BGE 向量 + BM25 + RRF | Hit@5 **0.67** |
| 服务化 | FastAPI + Redis 缓存 | 二次请求提速 **12.9×** |
| 面试题生成 | LLM 4 类 8 道/条 | 30 个岗位题库 |

---

## 快速体验

### 1. 环境准备

```bash
python -m pip install -r requirements.txt
```

### 2. 下载 BGE 模型（首次运行自动下载，约 100MB）

```bash
# 设置 HuggingFace 国内镜像（推荐）
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 3. 启动服务

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

启动后访问 `http://127.0.0.1:8001/docs` 查看 API 文档。

### 4. 试一下

```bash
# 检索岗位
curl -X POST http://127.0.0.1:8001/match \
  -H "Content-Type: application/json" \
  -d '{"query": "武汉的AI开发岗", "top_k": 5}'

# 查看某岗位的面试题
curl http://127.0.0.1:8001/interview/2384397
```

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        API 层（FastAPI）                     │
│  /health   /match   /interview/{job_id}   /cache            │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│  Redis     │  │ JobSearcher│  │ 面试题库    │
│  缓存      │  │  混合检索   │  │ (jsonl)    │
└────────────┘  └──────┬─────┘  └────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌────────────┐            ┌────────────┐
   │ 向量检索    │            │ BM25 检索  │
   │ BGE + Max  │            │ jieba      │
   └─────┬──────┘            └─────┬──────┘
         │                         │
         └────────────┬────────────┘
                      ▼
              ┌──────────────┐
              │  RRF 融合     │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  元数据过滤   │
              │ 城市/学历     │
              └──────────────┘
```

### 检索链路

1. **查询解析**：从 query 抽出城市/学历约束
2. **硬过滤**：先用约束过滤 JD 子集
3. **双路检索**：
   - 向量路：BGE-small-zh 编码，chunk 级相似度，job 级取 Max
   - BM25 路：jieba 分词 + 关键词精确匹配
4. **RRF 融合**：两路排名用 RRF（K=60）融合
5. **返回 Top-K**

---

## 关键实验结果

### 数据抽取（D2）

| 字段 | 规则版 | LLM 版 | 提升 |
| --- | --- | --- | --- |
| 城市 | 70% | **100%** | +30 |
| 学历 | 45% | **97%** | +52 |
| 技能关键词 | 65% | **100%** | +35 |
| 职责原文 | 70% | 80% | +10 |
| **整体** | — | **94%** | — |

### 检索（D3-D4）

| 版本 | 方法 | Hit@5 | Hit@10 | MRR |
| --- | --- | --- | --- | --- |
| v0 | TF-IDF | 0.5000 | 0.6667 | 0.3251 |
| v1 | BGE 向量 | 0.5333 | 0.6833 | 0.4048 |
| v2 | +硬过滤 | 0.5333 | 0.7000 | 0.4161 |
| v3 | +BM25 | **0.6667** | **0.7667** | **0.4615** |

### 服务化（D5）

| 请求 | 耗时 |
| --- | --- |
| 首次（无缓存） | 31.30 ms |
| 二次（命中缓存） | 2.43 ms |
| **加速比** | **12.9×** |

---

## 项目结构

```
求职AI工作台/
├── README.md                    # 本文件
├── DATA_SOURCES.md              # 数据来源与合规声明
├── resume.txt                   # 个人简历（用于匹配）
│
├── app.py                       # FastAPI 服务
├── search.py                    # 检索模块（JobSearcher 类）
├── requirements.txt
│
├── whpu_spider.py               # 爬虫（列表页）
├── whpu_detail_spider.py        # 爬虫（详情页）
├── data_clearner.py             # 规则版清洗
├── d2_llm_extract.py            # LLM 结构化抽取
├── d3_split.py                  # 切分
├── d3_embed.py                  # 向量化
├── d3_retrieve.py               # v1 检索
├── d4_filter.py                 # v2 硬过滤
├── d4_hybrid.py                 # v3 混合检索
├── d6_gen_qs.py                 # 面试题生成
│
├── eval/
│   ├── queries.jsonl            # 60 条评测集
│   ├── report.md                # 完整实验报告
│   └── jd_index.csv
│
└── data/
    ├── jd_llm_fixed.jsonl       # 292 条 LLM 结构化数据
    ├── jd_chunks.jsonl          # chunk 切片
    ├── embeddings.npz           # 向量缓存
    ├── match_v1.csv             # v1 检索结果
    ├── match_v2.csv             # v2 检索结果
    ├── match_v3.csv             # v3 检索结果
    └── interview_qs.jsonl       # 面试题库
```

---

## 数据合规

本项目的所有数据都来自**公开页面**，遵循以下规则（见 `DATA_SOURCES.md`）：

1. **白名单采集**：每条数据记录来源 URL 与采集时间
2. **尊重 robots.txt**：单线程、限频 1–2 秒/请求；UA 如实标识
3. **剔除个人身份字段**：只保留岗位描述类文本
4. **仅个人学习使用**：不对外发布、不商用、不售卖

**绝对不碰**：
- 需要登录才能看的页面
- 出现验证码、加密参数、签名接口的站点
- 简历库、候选人信息等个人隐私数据
- 多线程轰炸、代理池、高频请求

---

## 已知短板

- **数据层**：公司性质（国企/上市/私企）未进 chunk，影响部分查询
- **检索层**：BGE-small 对前后端/软硬件反义词区分不够，靠 BM25 弥补
- **服务层**：Redis 用 Memurai Developer 版，10 天自动停一次
- **应用层**：题库仅覆盖 30 条目标岗位

**14 条未命中的 query 中，13 条归因于评测集问题或数据本身，不是检索架构。**

---

## 后续计划

- [ ] Docker 打包，K8s 部署（W3）
- [ ] Prometheus 监控（W3）
- [ ] Query 改写（LLM 扩写宽泛查询）
- [ ] 公司性质字段进 chunk
- [ ] 前端界面（Streamlit / Gradio）

---

## 作者

骆宇成 · 2027 届 · 求职方向：AI 应用开发 / 后端开发 / IT 数字化

**项目周期**：2026.09.18 – 至今
**W2 实验报告**：见 [`eval/report.md`](eval/report.md)

---

## 引用

如果这个项目对你有帮助，欢迎 Star 或 Fork。

**技术栈**：
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [FastAPI](https://fastapi.tiangolo.com/)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)
- [DrissionPage](https://github.com/g1879/DrissionPage)