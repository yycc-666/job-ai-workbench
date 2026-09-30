# W2–W3 实验报告：从 TF-IDF 到混合检索服务

**周期**：2026.09.21 – 2026.09.30
**目标**：把 v0 TF-IDF 匹配升级为完整 RAG 检索系统，并服务化；后续完成容器化与 K8s 部署

> 本文第 0～8 章是 W2 的检索实验；末尾 W3D2～W3D4 是随后的容器化、K8s 部署与监控记录。

## 目录

- [0. 项目背景](#0-项目背景)
- [1. 实验总览](#1-实验总览)
- [2. D2：LLM 结构化抽取 vs 规则版](#2-d2llm-结构化抽取-vs-规则版)
- [3. D3：v0 vs v1 检索实验](#3-d3v0-vs-v1-检索实验)
- [4. D4：硬过滤 + BM25 混合检索](#4-d4硬过滤--bm25-混合检索)
- [5. D5：检索服务化（FastAPI + Redis）](#5-d5检索服务化fastapi--redis)
- [6. D6：面试题生成](#6-d6面试题生成)
- [7. 核心发现与已知短板](#7-核心发现与已知短板)
- [8. 面试要点速查](#8-面试要点速查)

W3 部署记录（追加章节）：

- [W3D2：容器编排（docker-compose）](#w3d2容器编排docker-compose)
- [W3D3：K8s 部署](#w3d3k8s-部署)
- [W3D4：Prometheus 监控](#w3d4prometheus-监控)

---

## 0. 项目背景

- **数据源**：武汉轻工大学就业信息网（公开页面，合规采集）
- **数据量**：292 条职位 JD
- **目标**：做一个自己每天用的求职工具——输入查询，输出匹配岗位 + 面试题
- **评测集**：60 条人工标注 query，覆盖城市、岗位方向、学历、技能、公司类型、模糊查询 6 类

---

## 1. 实验总览

### 四版检索对比（60 条评测 query）

| 指标 | v0 TF-IDF | v1 BGE 向量 | v2 +硬过滤 | v3 +BM25混合 |
| --- | --- | --- | --- | --- |
| Hit@5 | 0.5000 | 0.5333 | 0.5333 | **0.6667** |
| Hit@10 | 0.6667 | 0.6833 | 0.7000 | **0.7667** |
| Recall@5 | 0.3225 | 0.3539 | 0.3678 | **0.4647** |
| Recall@10 | 0.4428 | 0.4658 | 0.4992 | **0.5828** |
| MRR | 0.3251 | 0.4048 | 0.4161 | **0.4615** |

### 关键结论

- **BM25 混合检索是最大单步提升**：Hit@5 从 0.53 到 0.67（+25%）
- **硬过滤单独收益有限**（Hit@5 +0.00），但修复了城市约束的架构问题
- **MRR 从 0.33 到 0.46**，说明相关 JD 排序明显靠前

---

## 2. D2：LLM 结构化抽取 vs 规则版

### 2.1 实验设置

- 输入：`whpu_jobs_detail.csv`（292 条）
- 模型：小米 MiMo `mimo-v2.5-pro`
- 温度：0
- 抽取字段：岗位名 / 公司 / 城市 / 学历要求 / 技能关键词 / 职责原文
- 抽样：30 条，随机种子 42

### 2.2 准确率对比

| 字段 | 规则版 | LLM 版 | 提升 |
| --- | --- | --- | --- |
| 城市 | 70% | **100%** | +30 |
| 学历 | 45% | **97%** | +52 |
| 技能关键词 | 65% | **100%** | +35 |
| 职责原文 | 70% | **80%** | +10 |
| 岗位名 | 100% | **93%** | -7 |

**整体**：141/150 = 94%

### 2.3 判定规则

- 城市：LLM 抽的"XX市"是否出现在原始"工作地址"字段里
- 学历：从"招聘标签"（salary）里抽到的学历是否一致
- 技能：人工判断是否合理、无幻觉
- 职责：LLM 抽的段落是否是 JD 中"岗位职责"段
- 岗位名：是否保留核心信息（去届别、地点备注算对）

### 2.4 LLM 版典型错误

**错误 1：职责原文表头污染（6/30）**

LLM 把输入里的"标题：/公司：/工作地址："表头照抄进职责原文。
- 根因：prompt 未明确禁止
- 修法：已人工修正 6 条；如重跑，prompt 应加"职责原文禁止包含标题/公司/工作地址等表头字段"

**错误 2：学历歧义（1/30）**

`1857727` Java开发工程师：salary 字段为"不限本科"，"不限"指工作经验，"本科"才是学历，LLM 误抽成"不限"。
- 修法：已从 salary 字段补齐，标记 `学历来源: salary标签补齐`

**错误 3：岗位名信息丢失（2/30）**

LLM 按"简短"规则重写，丢掉了"（C++/Go方向）""（Python、C++方向）"等技术方向。
- 结论：岗位名不该让 LLM 改写，应直接复用 title

### 2.5 结论

1. LLM 在城市、学历、技能三个字段上提升显著（+30/+52/+35），达到可用水平
2. 职责原文提升有限（+10），主要瓶颈是 prompt 未处理表头污染
3. 岗位名适合直接复用 title，无需 LLM 改写
4. **最终产物**：`data/jd_llm_fixed.jsonl`（292 条，学历已补齐，6 条职责已人工修正）

---

## 3. D3：v0 vs v1 检索实验

### 3.1 方法

- **v0**：整条 JD 文本 → TF-IDF 向量 → 余弦相似度
- **v1**：每条 JD 切 3 类块（元信息/技能/职责），BGE-small-zh 向量化，job 级取 Max

### 3.2 切分策略

- 元信息块：岗位名 + 公司 + 城市 + 学历
- 技能块：技能关键词
- 职责块：按 300 字切，overlap 50 字
- 一条 JD 平均 4–8 块

### 3.3 指标对比

| 指标 | v0 TF-IDF | v1 BGE | 相对提升 |
| --- | --- | --- | --- |
| Hit@5 | 0.5000 | 0.5333 | +6.7% |
| Hit@10 | 0.6667 | 0.6833 | +2.5% |
| Recall@5 | 0.3225 | 0.3539 | +9.7% |
| Recall@10 | 0.4428 | 0.4658 | +5.2% |
| **MRR** | 0.3251 | 0.4048 | **+24.5%** |

### 3.4 解读

v1 在 MRR 上提升显著（+24.5%），说明向量检索让相关 JD 的**排序更靠前**；
但 Hit@5/Recall@5 提升有限（+2~10%），说明**召回率瓶颈不在 embedding，在检索架构**。

### 3.5 v1 未命中归因（19 条）

| 类型 | 条数 | 根因 | 是否 D4 解决 |
| --- | --- | --- | --- |
| 城市硬约束缺失 | 4 | 语义分不清"东莞"vs"武汉" | ✅ D4 硬过滤 |
| 公司属性未进 chunk | 2 | "国企/制造业"没进 meta | ✅ D4 硬过滤 |
| 反义词/方向混淆 | 3 | BGE 不懂"前后端"对立 | ✅ D4 混合检索 |
| query 太宽泛 | 5 | 相关 JD 十几条，Top10 排不下 | ❌ 评测集问题 |
| 数据集稀疏 | 5 | 相关 JD 只有 1–2 条 | ❌ 数据问题 |

### 3.6 D4 三个改进方向

1. **元数据硬过滤**：从 query 识别城市/学历/公司性质，先过滤再检索
2. **混合检索**：BM25 + 向量，解决反义词/关键词精确匹配
3. **query 改写**：LLM 把宽泛 query 扩写

---

## 4. D4：硬过滤 + BM25 混合检索

### 4.1 方法

- **v2**：从 query 识别城市/学历约束，先过滤 JD 子集，再向量检索
- **v3**：v2 + BM25 关键词检索，RRF 融合（K=60，向量/BM25 权重各 1.0）

### 4.2 指标对比

| 指标 | v1 | v2 | v3 | v3-v2 |
| --- | --- | --- | --- | --- |
| Hit@5 | 0.5333 | 0.5333 | **0.6667** | +0.1333 |
| Hit@10 | 0.6833 | 0.7000 | **0.7667** | +0.0667 |
| Recall@5 | 0.3539 | 0.3678 | **0.4647** | +0.0969 |
| Recall@10 | 0.4658 | 0.4992 | **0.5828** | +0.0836 |
| MRR | 0.4048 | 0.4161 | **0.4615** | +0.0454 |

### 4.3 关键发现

1. **硬过滤（v2）单独提升有限（+0.00 Hit@5），但解决了城市约束的架构问题**
2. **BM25 混合检索（v3）是最大单步提升（Hit@5 +25%），修复了 6 条关键词匹配场景**
3. **14 条未命中已归因，其中 13 条不在检索射程内**

### 4.4 v3 修好的 6 条（BM25 关键贡献，按 Top-10 统计）

- q003 东莞的研发岗位
- q011 后端开发岗位
- q020 客户端开发岗位
- q030 武汉销售类本科校招
- q033 需要 C 语言开发岗位
- q054 东莞机器视觉本科27届

### 4.5 v3 改坏的 2 条（按 Top-10 统计）

- **q031 需要会Python的岗位**：BGE 对"Python"语义不敏感，BM25 也没救
- **q043 国企里的IT技术岗**：公司属性未进 chunk，三版都无解

> **口径说明**：上面 4.4、4.5 两份清单是**按 Top-10 统计**的——评测脚本 `d4_eval_v3.py` 里判定命中的切片写的是 `[:10]`。它与 Hit@10 的变化吻合：0.7000 → 0.7667，净增 4 条 = 6 − 2。
>
> 如果按 Top-5 统计，修好的是 10 条、改坏的是 2 条（q001、q015），与 Hit@5 的 +0.1333（≈8 条）吻合。**两套都正确，只是量的不是同一件事**——Top-10 上的"改坏"是召回退步（掉出前 10），Top-5 上的"改坏"是排序退步（仍在榜上，但从前 5 掉到了 6～10 名）。

### 4.6 v3 仍未命中归因（14 条）

| 类型 | 条数 |
| --- | --- |
| query 太宽泛（评测集问题） | 5 |
| 公司属性未进 chunk（数据问题） | 4 |
| 数据稀疏（期望 1-2 条） | 3 |
| BGE 分词局限 | 1 |
| 排序边缘问题 | 1 |

### 4.7 面试要点

- **硬过滤**：为什么检索要加元数据约束
- **BM25 + 向量混合**：为什么纯向量不够，RRF 融合的原理
- **RRF 常数 K=60**：为什么是 60（经典推荐值，平衡头部和尾部）
- **指标提升的解读**：Hit@5 提升 25% 的真正原因

---

## 5. D5：检索服务化（FastAPI + Redis）

### 5.1 方法

- `search.py`：检索逻辑封装为 `JobSearcher` 类，模型和数据在服务启动时加载一次
- `app.py`：FastAPI 服务，初始 3 个接口，D6 后扩展到 4 个（W3D4 再加 `/metrics`，共 5 个）
  - `GET /health`：健康检查
  - `POST /match`：检索接口
  - `GET /interview/{job_id}`：面试题接口（D6 新增）
  - `DELETE /cache`：清理缓存（2026-09-30 起改为只清 `match:` 前缀，不再用 `flushdb()`）
- Redis 缓存：key = `md5(query|top_k)`，TTL = 1 小时，异常时降级为无缓存

### 5.2 验证结果

| 检查项 | 结果 |
| --- | --- |
| 模型加载次数 | 启动时 1 次，后续请求不重载 |
| `/health` 返回 | `{"status":"ok","jd_count":292,"cache":"on"}` |
| `/match` 首次请求 `cached` | `false` |
| `/match` 二次请求 `cached` | `true` |

### 5.3 性能对比（query = "武汉的Python岗"，top_k=5）

| 请求 | 耗时 |
| --- | --- |
| 第一次（无缓存，含 BGE 编码 + 向量检索 + BM25 + RRF 融合） | **31.30 ms** |
| 第二次（命中缓存，直接返回） | **2.43 ms** |
| **加速比** | **约 12.9 倍** |

### 5.4 优雅降级

- Redis 未启动时：服务照常工作，`/health` 返回 `cache: "off"`
- Redis 中途挂掉：读写异常被 try/except 捕获，检索不中断
- **连得上但不返回**（2026-09-30 补）：原来只设了 `socket_connect_timeout`、没设 `socket_timeout`，这种场景下请求会**永久挂住**。现在两个超时都设为 2 秒，并显式配置 `retry=Retry(NoBackoff(), 0)`——因为 redis-py 8.x 的默认策略是重试 10 次，"删掉弃用的旧参数"反而会让请求挂得更久

### 5.5 面试要点

1. **为什么服务化**：脚本方式每次运行都要重载 BGE 模型（本机 Windows 热启动约 5–10 秒，容器冷启动 90 秒以上）；服务化后模型常驻内存，只在启动时加载一次，请求响应进入毫秒级
2. **为什么加缓存**：query → 结果映射重复计算浪费算力，加 Redis 缓存后重复请求从 31 ms 降到 2.4 ms
3. **key 设计**：`md5(query|top_k)` —— md5 避免特殊字符，加 `top_k` 防止不同深度互相覆盖
4. **降级设计**：Redis 是加速手段，不是必需依赖，挂了不影响核心检索功能

---

## 6. D6：面试题生成

### 6.1 方法

- `d6_gen_qs.py`：批量调用 LLM，对每条 JD 生成 8 道面试题（4 类各 2 道）
- **Prompt 关键规则**：
  - 技术基础题只看 JD 技能，禁止拿简历技能当考点
  - 项目深挖题从简历项目出题
  - 岗位匹配题问动机和迁移
  - 行为面试题问通用软技能
- `/interview/{job_id}` 接口：输入 job_id 返回题目

### 6.2 产出

- `data/interview_qs.jsonl`：30 条目标岗位的题库
- `GET /interview/{job_id}` 接口，服务端启动时加载题库

### 6.3 质量验证

抽查 5 条 AI 应用开发岗（2384397、2384848、2385692、2391103、2385096）：

| JD | 技术基础题 | 判断 |
| --- | --- | --- |
| 信锐 AI应用开发 | 监督/无监督学习、RAG | ✅ 对口 |
| 先导 AI应用开发 | LLM理解、Prompt设计、RAG | ✅ 对口 |
| 奥普特 AI应用研发 | RAG原理、大模型API封装 | ✅ 对口 |
| 宇鲸 AI全栈 | FastAPI、LangChain编排 | ✅ 对口 |
| 方阵 AI全栈 | LLM集成、API稳定性 | ✅ 对口 |

### 6.4 修复记录

初版 prompt 中「技术基础」题严重跑偏，结构工程师岗问爬虫、零售岗问 BeautifulSoup。
修改 prompt 后加规则「技术基础题只看 JD 技能，禁止拿简历技能当考点」，三条测试全部修对。

### 6.5 隐含价值

如果某条 JD 的"岗位匹配"题让我**答不上来**，说明这个岗位不该投。
它帮我提前识别"不匹配风险"——这是搜索本身做不到的。

### 6.6 遗留问题

- 题库只覆盖 30 个目标岗位，非全量 292 条
- LLM 生成的题目偶尔有重复，需要人工过滤

---

## 7. 核心发现与已知短板

### 7.1 三个非直觉发现

1. **硬过滤单独收益有限，但架构上必要**：v2 的 Hit@5 提升是 +0.00，但它让"城市约束"变成可解释、可复现的机制，而不是靠 embedding 撞运气
2. **BM25 补上了向量的最大短板**：向量懂"语义相近"，BM25 懂"关键词精确匹配"，两者互补
3. **14 条未命中里 13 条不在检索射程内**：评测集问题、数据问题不能靠检索架构解决

### 7.2 已知短板

- **数据层**：公司性质（国企/上市/私企）未进 chunk，导致 q041/q043/q044 无法区分
- **检索层**：BGE-small 对前后端/软硬件反义词区分不够
- **服务层**：Redis 用 Memurai Developer 版，10 天自动停一次
- **应用层**：题库仅覆盖 30 条目标岗位，非全量

---

## 8. 面试要点速查

**数据层**：LLM 抽取整体 94%，学历从 45% 提到 97%；岗位名不该让 LLM 改写，直接复用 title。

**检索层**：v0→v3 走完四个版本，Hit@5 从 0.50 到 0.67；BM25 是最大单步提升；14 条未命中里 13 条归因于数据和评测集，不是检索架构。

**服务层**：FastAPI + Redis 缓存，二次请求提速 12.9 倍；优雅降级，Redis 挂了不影响检索。

**应用层**：LLM 生成 8 道/条面试题；核心价值不是生成题目，而是帮自己识别"不匹配岗位"。

## W3D2：容器编排（docker-compose）

### 方法

- `docker-compose.yml` 定义两个服务：
  - `redis`：官方 `redis:7-alpine`，启用 healthcheck
  - `app`：`job-search:0.3`，通过 `REDIS_HOST=redis` 连接
- `depends_on: service_healthy` 保证 Redis 就绪后再启动 app

### 验证结果

| 检查项 | 结果 |
| --- | --- |
| `docker compose up -d` | 3 步成功 |
| `job-redis` 状态 | Healthy |
| `job-app` 状态 | Up (healthy) |
| 启动日志 | Redis 连接成功 |
| `/health` 返回 | `{"status":"ok","jd_count":292,"cache":"on"}` |
| 两次 `/match` cached | `false` → `true` |

### 性能对比（容器环境）

| 请求 | 耗时 |
| --- | --- |
| 第一次（无缓存） | 58.36 ms |
| 第二次（命中缓存） | 15.95 ms |
| **加速比** | **3.7 倍** |

### 本地 vs 容器对比

| 环境 | 第一次 | 第二次 | 加速比 |
| --- | --- | --- | --- |
| W2 本地（uvicorn 直跑） | 31.30 ms | 2.43 ms | 12.9× |
| W3 容器（docker-compose） | 58.36 ms | 15.95 ms | 3.7× |

**解读**：容器环境带来额外开销（CPU 共享、网络转发），但缓存依然显著。
生产环境讲响应时间，必须说明是哪种部署方式。

### 面试要点

1. **为什么用 docker-compose**：一条命令起全栈，环境一致，避免"我本地能跑"
2. **服务发现**：容器间通过服务名（`redis`）通信，不是 `127.0.0.1`
3. **健康检查**：`depends_on: service_healthy` 保证启动顺序，不是盲目 `depends_on`
4. **端口策略**：Redis 不映射宿主端口，避免和本机 Memurai 冲突

## W3D3：K8s 部署

> **2026-09-30 更新**：本节最初记录的是在 Docker Desktop 自带 Kubernetes 上的部署。
> 后来为了让新镜像上线，发现该集群的节点容器在宿主机上不可见（`docker ps` 查不到），
> 无法 `docker exec` 进去导入本地镜像，因此**改用自建的 kind 集群**重新部署。

### YAML 文件

- `k8s/redis-deployment.yaml`：Redis 部署，含 liveness/readiness probe
- `k8s/redis-service.yaml`：ClusterIP 服务
- `k8s/app-deployment.yaml`：app 部署，2 副本，`imagePullPolicy: Never`
- `k8s/app-service.yaml`：NodePort 服务，30080

### 部署环境

| 项 | 值 |
| --- | --- |
| 集群 | Docker Desktop 自带 Kubernetes（初版）→ 自建 kind 集群（2026-09-30 起） |
| 节点 | `desktop-control-plane`（v1.34.3）→ `demo-control-plane`（v1.37.0） |
| 命名空间 | `default`（初版）→ `job-search`（2026-09-30 起） |

换集群的原因和换命名空间的原因都写在下面「踩过的坑」里。

### 部署结果

| 检查项 | 结果 |
| --- | --- |
| Pod 数量 | 3（2 app + 1 redis） |
| Pod 状态 | 全部 `1/1 Running`，`RESTARTS 0` |
| Service 后端 | app: 2 endpoints；redis: 1 endpoint |
| 新 Pod 就绪耗时 | 30 秒（改用 `startupProbe` 之后） |

### 关键设计

- **服务发现**：app 通过 `REDIS_HOST=redis` 走 K8s DNS 解析到 redis Service
- **多副本**：`replicas: 2` 演示水平扩展
- **本地镜像**：`imagePullPolicy: Never`，镜像需提前进入节点容器的 containerd（见坑 1）
- **启动探测**：用 `startupProbe`（10 秒 × 18 次 = 最长 180 秒），而不是给 liveness 设 `initialDelaySeconds`（见坑 2）

### 验证结果

- `/health`：`{status:ok, jd_count:292, cache:on}`
- `/match`：返回 3 条武汉 AI 岗（2384112、2376253、2369258）
- **负载均衡**：10 次请求，两个 pod 各处理 5 次
- **日志格式**（2026-09-30 起）：带时间戳、级别与 logger 名，例：

  ```
  2026-09-30 09:16:35 INFO [job-search] 启动完成，共加载 292 条 JD
  ```

### 踩过的坑

#### 坑 1：本地镜像进不了集群（2026-09-30 定位）

**最初以为**：镜像需要手动用 `ctr images import` 导入 K8s 节点的 containerd。

**实际原因**：Docker Desktop 的 Kubernetes **节点容器在宿主机上不可见**——`docker ps -a` 里查不到 `desktop-control-plane`，所以无法 `docker exec` 进去执行导入。改了镜像 tag（0.2 → 0.3）之后，新 Pod 一直卡住：

```
Warning  ErrImageNeverPull  kubelet
  Container image "job-search:0.3" is not present with pull policy of Never
```

**排查的关键一步**是 `kubectl describe pod` 里的这一行：

```
Successfully assigned default/job-app-... to desktop-control-plane
```

它说明 Pod 跑的节点，和我执行导入的那个容器（`demo-control-plane`）**根本不是同一个**。

**试过的替代方案**：

| 方案 | 结果 |
| --- | --- |
| 推镜像到 Docker Hub，让 k8s 自己拉 | ❌ `registry-1.docker.io` / `hub.docker.com` 连接超时 |
| 改用自建的 kind 集群（节点容器可见） | ✅ 成功 |

**最终做法**：导出 kind 节点的 kubeconfig、把 `server` 改成宿主机的映射端口；用独立命名空间 `job-search` 部署（该集群里已有一个叫 `redis` 的 Deployment，避免撞名）。

**结论**：这个机制**依赖 Docker Desktop 的内部实现，不可复现**。更通用的做法是推送到镜像仓库由 k8s 拉取——本次因网络受限未能采用。操作步骤已完整记录，不再依赖"记忆中的做法"。

#### 坑 2：`startupProbe` 与 `initialDelaySeconds`（2026-09-30 修正）

**原来的配置**：只有 liveness 和 readiness，靠 `initialDelaySeconds: 90` 等启动完成。

**问题**：BGE 模型加载实测就要 90 秒以上，**两个数字卡在边界上**。机器稍慢或镜像冷启动，启动就会超过 90 秒，liveness 开始失败并杀掉容器；重启后又加载模型、又超时——进入 CrashLoopBackOff，而日志里只看到"Pod 反复重启"。

**改后**：

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  periodSeconds: 10
  failureThreshold: 18      # 10 × 18 = 最长 180 秒
```

`startupProbe` 成功之前，liveness 和 readiness **都不生效**，所以启动慢不会再被误杀；启动成功后 liveness 又能立刻正常监护。加了它之后，两个探针的 `initialDelaySeconds` 就删掉了——启动窗口由 startup 负责。

**效果**：Ready 的**理论下限从 60 秒降到 10 秒**（前者由 readiness 的 `initialDelaySeconds` 决定）。实测新 Pod 30 秒就绪。

> 注：30 秒是在节点已热的情况下观测到的，与 9 天前的冷启动不可直接对比；**"下限降低"才是配置层面确定的结论**。

#### 坑 3：其他

- **containerd image store**：Docker Desktop 的 Kubernetes 依赖 containerd 后端，需在设置中确认启用
- **NodePort 在本地集群里从宿主访问不到**：30080 需要在建集群时配置 `extraPortMappings`，否则只能用 `kubectl port-forward`（这也是本次全程用的方式）

## W3D4：Prometheus 监控

### 方法

- `app.py` 加 5 个 Prometheus 指标：
  - `match_requests_total{cached}`：Counter，按是否命中缓存打标签
  - `match_latency_seconds`：Histogram，桶 [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
  - `interview_requests_total{status}`：Counter，按 found/not_found 打标签
  - `cache_hits_total`：Counter
  - `app_resident_memory_bytes`：Gauge，进程常驻内存（2026-09-30 新增；读 `/proc/self/status` 的 `VmRSS`，用于校准 K8s 的 `requests`）
- `/metrics` 端点用 `prometheus_client.generate_latest()` 输出
- Prometheus 容器通过 `host.docker.internal:8888` 抓取 port-forward 后的服务

### 验证结果

| 检查项 | 结果 |
| --- | --- |
| Prometheus Target | job-app UP |
| `match_requests_total` | cached=false: 1, cached=true: 6 |
| `cache_hits_total` | 6 |
| `rate(match_requests_total[1m])` | 0（过去 1 分钟无新请求） |
| `histogram_quantile(0.95, ...)` | NaN（样本少）→ 发更多流量后可计算 |

### 关键设计

- **Counter vs Histogram**：Counter 统计总量，Histogram 统计延迟分布
- **标签设计**：`cached=true/false` 能区分有/无缓存请求；`status=found/not_found` 能区分查询成功/失败
- **桶选择**：延迟桶覆盖 10ms–5s，是这类 Web 服务的典型延迟范围
- **抓取间隔**：15 秒，平衡实时性和 Prometheus 负载

### 面试要点

1. **为什么用 Prometheus**：拉模式（pull），服务只需暴露 `/metrics`，Prometheus 定期抓取
2. **四种指标类型**：Counter（只增）、Gauge（可增可减）、Histogram（分布）、Summary（分位数）
3. **Histogram vs Summary**：Histogram 分位数在服务端算（可聚合），Summary 在客户端算（不可聚合）
4. **PORT-forward vs NodePort**：Docker Desktop 的 K8s 里 NodePort 不通，port-forward 是最稳的临时访问方式