import os #读取环境变量，避免把本机路径写死在代码里
from pathlib import Path
from DrissionPage import ChromiumPage, ChromiumOptions  #模拟网站 获取网站信息
from bs4 import BeautifulSoup #翻译HTML文档
import csv 
import time

def create_page():
    co = ChromiumOptions()
    #浏览器所在位置（可用环境变量 WHPU_EDGE_BROWSER 覆盖）
    co.set_browser_path(os.getenv("WHPU_EDGE_BROWSER", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"))
    #给这个模拟浏览器单独开一个用户文件夹（放到项目外，避免 Cookies / 登录数据混进仓库）
    co.set_user_data_path(os.getenv("WHPU_EDGE_PROFILE", str(Path.home() / ".whpu_edge_profile")))
    co.set_argument('--disable-extensions')
    return ChromiumPage(co)

def get_job_detail(page, url):
    page.get(url) #命令启动的浏览器访问指定网址
    time.sleep(2)  # 等待详情页 JS 渲染
    
    html = page.html #将当前浏览器页面已经渲染好的完整HTML源码取出来，赋值给变量html
    soup = BeautifulSoup(html, "html.parser") #把网页源码字符串，解析成一颗可以查找的数
    
    # 初始化一个字典存放详情数据
    detail = {
        "职能类别": "", "招聘人数": "", "工作经验": "", "语言要求": "",
        "需求专业": "", "工作地址": "",
        "公司名称": "", "单位性质": "", "单位行业": "", "单位规模": "",
        "职位详情": ""
    }
    
    # 1. 提取基础信息（遍历 ul.clearfix 下的 li）
    info_items = soup.select(".details-list ul.clearfix li")
    for item in info_items:
        text = item.get_text(strip=True)
        span = item.find("span")
        value = span.get_text(strip=True) if span else ""#如果span存在，提取文字，如果不存在，使用空字符串
        
        if "职能类别" in text: detail["职能类别"] = value
        elif "招聘人数" in text: detail["招聘人数"] = value
        elif "工作经验" in text: detail["工作经验"] = value
        elif "语言要求" in text: detail["语言要求"] = value
        # 隐私：不采集联系人 / 联系电话等个人身份字段，只保留岗位描述类信息
        elif "需求专业" in text: detail["需求专业"] = value

    # 2. 提取工作地址
    address_tag = soup.select_one(".details-address .address-container")
    if address_tag:
        detail["工作地址"] = address_tag.get_text(strip=True, separator="\n")

    # 3. 提取公司信息（侧边栏 siderinfo）
    company_tag = soup.select_one(".siderinfo .unit-info a")
    if company_tag:
        detail["公司名称"] = company_tag.get_text(strip=True)
    
    sidebar_items = soup.select(".siderinfo .info .item")
    for item in sidebar_items:
        label_tag = item.select_one(".label")
        value_tag = item.select_one("span")
        if label_tag and value_tag:
            label = label_tag.get_text(strip=True)
            value = value_tag.get_text(strip=True)
            if "单位性质" in label: detail["单位性质"] = value
            elif "单位行业" in label: detail["单位行业"] = value
            elif "单位规模" in label: detail["单位规模"] = value

    # 4. 提取职位详情（截图未展开，我们尝试抓取 details-mge 里的内容）
    desc_tag = soup.select_one(".details-mge")
    if desc_tag:
        detail["职位详情"] = desc_tag.get_text(strip=True, separator="\n")

    return detail

# ===== 主程序 =====
# 1. 读取昨天保存的 whpu_jobs.csv
jobs = []
# 输入是列表页爬虫产出的本地中间文件（含联系方式，禁止提交）
with open(os.getenv("WHPU_JOBS_CSV", "whpu_jobs.csv"), "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        jobs.append(row)

print(f"共读取 {len(jobs)} 条职位，开始抓取详情页...")

# 2. 启动浏览器，遍历抓取
page = create_page()
all_details = []

# 先只抓前 5 条测试，没问题后把 [:5] 去掉即可抓全部
for i, job in enumerate(jobs): 
    print(f"[{i+1}/{len(jobs)}] 正在抓取: {job['title']}")
    try:
        detail = get_job_detail(page, job["url"])
        # 把列表页的基础信息 + 详情页信息合并
        job.update(detail)
        all_details.append(job)
        print(f"  -> 成功获取，公司: {detail['公司名称']}")
    except Exception as e:
        print(f"  -> 抓取失败: {e}")
    time.sleep(1)  # 礼貌间隔，防封

page.quit()

# 3. 保存到新的 CSV 文件
# 注意：输出文件含联系人 / 联系电话，属于他人个人信息，禁止提交（.gitignore 已排除）
with open(os.getenv("WHPU_JOBS_DETAIL_CSV", "whpu_jobs_detail.csv"), "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "title", "company", "salary", "pub_date", "url",  # 列表页原有字段
        "职能类别", "招聘人数", "工作经验", "语言要求", 
        "需求专业", "工作地址",
        "公司名称", "单位性质", "单位行业", "单位规模", "职位详情"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_details)

print(f"\n测试抓取完成！共抓取 {len(all_details)} 条详情数据，已保存到 whpu_jobs_detail.csv")