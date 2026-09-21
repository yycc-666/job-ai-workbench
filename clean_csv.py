import csv
import re

def parse_salary(text):
    """从薪资文本中拆出 薪资范围 / 地点 / 性质 / 学历"""
    result = {
        "薪资范围": "",
        "工作地点": "",
        "工作性质": "",
        "学历要求": ""
    }
    
    # 1. 提取薪资：匹配类似 7500-8499 的数字
    salary_match = re.search(r'(\d+-\d+)', text)
    if salary_match:
        result["薪资范围"] = salary_match.group(1)
    
    # 2. 提取地点：匹配 "XX省XX市" 或 "XX市"
    city_match = re.search(r'([\u4e00-\u9fa5]+省[\u4e00-\u9fa5]+市|[\u4e00-\u9fa5]+市)', text)
    if city_match:
        result["工作地点"] = city_match.group(1)
    
    # 3. 提取性质：全职 / 兼职 / 实习
    nature_match = re.search(r'(全职|兼职|实习)', text)
    if nature_match:
        result["工作性质"] = nature_match.group(1)
    
    # 4. 提取学历：本科 / 硕士 / 博士 / 大专 / 不限
    edu_match = re.search(r'(本科|硕士|博士|大专|不限)', text)
    if edu_match:
        result["学历要求"] = edu_match.group(1)
    
    return result

def parse_company(text):
    """尝试从公司文本中拆分出 公司名 / 行业 / 规模"""
    result = {
        "公司名称_清洗": text,
        "所属行业": "",
        "公司规模": ""
    }
    
    # 1. 提取规模：匹配 "10000人以上" 或 "50-150人"
    scale_match = re.search(r'(\d+人以上|\d+-\d+人)', text)
    if scale_match:
        result["公司规模"] = scale_match.group(1)
        # 把规模从公司名里删掉
        text = text.replace(scale_match.group(1), "")
    
    # 2. 提取行业：常见的行业后缀
    industry_keywords = ["金融业", "信息传输", "软件和信息技术服务业", "制造业", 
                         "科学研究和技术服务业", "租赁和商务服务业", "批发和零售业",
                         "农、林、牧、渔业", "教育", "文化、体育和娱乐业"]
    for kw in industry_keywords:
        if kw in text:
            result["所属行业"] = kw
            # 把行业从公司名里删掉
            text = text.replace(kw, "")
            break
    
    # 剩下的部分就是干净的公司名
    result["公司名称_清洗"] = text.strip()
    return result

# ===== 主程序 =====
rows = []
with open("whpu_jobs_detail.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 处理薪资列
        salary_info = parse_salary(row.get("salary", ""))
        row.update(salary_info)
        
        # 处理公司列
        company_info = parse_company(row.get("company", ""))
        row.update(company_info)
        
        rows.append(row)

# 保存到新文件，新增了拆分后的列
new_fields = list(reader.fieldnames) + ["薪资范围", "工作地点", "工作性质", "学历要求", "公司名称_清洗", "所属行业", "公司规模"]

with open("whpu_jobs_cleaned.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

print(f"清洗完成！共处理 {len(rows)} 条数据，已保存到 whpu_jobs_cleaned.csv")