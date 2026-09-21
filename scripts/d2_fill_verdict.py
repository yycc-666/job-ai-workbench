import csv

# 逐条核对后的判定（只记"错"的行，其余全"对"）
# 判定依据：LLM prompt 要求 岗位名简短、学历只取五选一、职责须以岗位职责/工作职责开头到任职要求前结束
wrong = {
    ("https://whpu.91wllm.cn/job/view/id/2386410", "职责"),  # 混入标题/公司/工作地址等表头
    ("https://whpu.91wllm.cn/job/view/id/2386847", "职责"),  # 同上
    ("https://whpu.91wllm.cn/job/view/id/2389420", "职责"),  # 同上
    ("https://whpu.91wllm.cn/job/view/id/1857727", "学历"),  # salary"不限本科"中"不限"是工作经验，学历实为本科
    ("https://whpu.91wllm.cn/job/view/id/2375073", "职责"),  # 混入表头
    ("https://whpu.91wllm.cn/job/view/id/2391238", "职责"),  # 混入表头+公司简介全文
    ("https://whpu.91wllm.cn/job/view/id/2385849", "岗位名"),  # 丢了（C++/Go方向）技术方向
    ("https://whpu.91wllm.cn/job/view/id/2384398", "职责"),  # 混入【职位亮点】【丰厚薪酬】等福利段
    ("https://whpu.91wllm.cn/job/view/id/2338523", "岗位名"),  # 输出"课程导师"，丢少儿编程/长期班/Python C++方向
}

# 填表前先验证 1857727 的工作经验字段，确认"不限"不是学历
with open("whpu_jobs_detail.csv", "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["url"] == "https://whpu.91wllm.cn/job/view/id/1857727":
            print(f"验证 1857727: salary={row['salary']!r} 工作经验={row['工作经验']!r}")
            break

path = "data/d2_compare30.csv"
rows = []
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

filled = 0
for row in rows:
    if len(row) >= 5 and row[0] and row[0] != "url":
        key = (row[0], row[1])
        row[4] = "错" if key in wrong else "对"
        filled += 1

with open(path, "w", encoding="utf-8-sig", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"已填写 {filled} 行判定，其中 {sum(1 for r in rows if len(r) >= 5 and r[0] and r[0] != 'url' and r[4] == '错')} 行判错")
