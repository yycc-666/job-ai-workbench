import os #读取环境变量，避免把本机路径写死在代码里
from pathlib import Path
from DrissionPage import ChromiumPage, ChromiumOptions #模拟网站 获取网站信息
from bs4 import BeautifulSoup #翻译HTML文档
from urllib.parse import urljoin #将相对路径拼成完整网址
import csv #写csv文件使用
import time #时间

base_url = "https://whpu.91wllm.cn" #网站的基础域名（根地址）
start_url = "https://whpu.91wllm.cn/job/search/domain/whpu/title//city/320000%2C330000%2C420000%2C430000%2C440000/d_skill//d_industry//d_major/08090000/d_education/101/d_category//d_salary//nature//scale//time//page/1"
#需要爬取的界面

def create_page():  #定义函数
    co = ChromiumOptions() #创建一份“浏览器启动配置”
    #浏览器所在位置（可用环境变量 WHPU_EDGE_BROWSER 覆盖）
    co.set_browser_path(os.getenv("WHPU_EDGE_BROWSER", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"))
    #给这个模拟浏览器单独开一个用户文件夹（放到项目外，避免 Cookies / 登录数据混进仓库）
    co.set_user_data_path(os.getenv("WHPU_EDGE_PROFILE", str(Path.home() / ".whpu_edge_profile")))
    co.set_argument('--disable-extensions') #不允许加载任何插件
    return ChromiumPage(co) #返回模拟的浏览器页面

def scrape_one_page(page, url, seen_urls): #定义函数
    print(f"\n正在抓取: {url}") #打印正在抓取第几页
    page.get(url) #命令浏览器去访问这个URL
    time.sleep(3) #等待三秒
    
    html = page.html #将当前浏览器界面中，已经渲染好的完整HTML源码，赋值给变量html
    soup = BeautifulSoup(html, "html.parser") #把网页源码字符串，解析成一颗“可以查找的数”
    
    jobs = [] #将爬取的job部分放入这个列表
    job_links = soup.find_all("a", href=lambda h: h and "/job/view/id/" in h) #条件过滤
    
    for link in job_links: #循环
        title = link.get_text(strip=True) #将字拿出来 去掉字前后空格
        href = link.get("href") #拿出这个标签的href属性值
        detail_url = urljoin(base_url, href) #拼成完整网址
        
        # 去重逻辑：如果这个链接已经抓过了，跳过
        if detail_url in seen_urls:
            continue
        seen_urls.add(detail_url) #seen_urls是集合set()。集合的特点就是不允许有重复元素。如果没有重复元素就加进去 如果有就会跳过循环 继续下一个
        
        parent = link.find_parent("div", class_="job") #找寻最近的符合要求的父标签
        if not parent: #如果没找到 随便找个div作为父亲。找到父标签是为了在这个范围内去搜索公司名等等
            parent = link.find_parent("div")
        
        company = ""  #初始化 定义一个空字符串
        salary = ""
        pub_date = ""
        
        if parent: #parent不是数组，而是一个HTML标签对象(Tag),当找到parent这个Tag就进行下面的操作
            company_tag = parent.select_one(".company")
            if company_tag:
                company = company_tag.get_text(strip=True)
            salary_tag = parent.select_one(".salary")
            if salary_tag:
                salary = salary_tag.get_text(strip=True)
            date_tag = parent.select_one(".name span")
            if date_tag:
                pub_date = date_tag.get_text(strip=True)
        
        jobs.append({ #jobs是一个列表，这句话是将下方用字典表示的东西装入列表中
            "title": title,
            "company": company,
            "salary": salary,
            "pub_date": pub_date,
            "url": detail_url
        })
    
    # 找下一页
    next_btn = soup.select_one(".pages li.next a") #CSS选择器。意思是在网页里找class=“pages”里面的li标签，并且li的class=“next” 再在li找<a>标签。 就是找下一页按钮，如果找到了，就用get("href")拿链接
    next_url = None
    if next_btn and next_btn.get("href"):
        next_url = urljoin(base_url, next_btn.get("href"))
        # ✅ 核心修复：如果下一页链接等于当前链接，说明到最后一页了
        if next_url == url:
            print("已到最后一页（下一页链接指向当前页）。")
            next_url = None
        else:
            print(f"找到下一页: {next_url}")
    else:
        print("没有下一页了。")
    
    return jobs, next_url

# ===== 主程序 =====
all_jobs = []
seen_urls = set()  # set()是python中的集合，有去重的功能
page = create_page()

current_url = start_url
page_num = 1

while current_url:
    jobs, next_url = scrape_one_page(page, current_url, seen_urls)
    all_jobs.extend(jobs)
    print(f"第 {page_num} 页完成，本页新增 {len(jobs)} 条，累计 {len(all_jobs)} 条。")
    
    if not next_url:
        break
    
    current_url = next_url
    page_num += 1
    time.sleep(1)

page.quit()

# 保存 CSV
with open("whpu_jobs.csv", "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = ["title", "company", "salary", "pub_date", "url"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_jobs)

print(f"\n全部完成！共 {len(all_jobs)} 条唯一职位，已保存到 whpu_jobs.csv")