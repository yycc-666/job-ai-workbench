import os
import requests

API_KEY = os.getenv("MIMO_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置环境变量 MIMO_API_KEY")

API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-pro"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "用一句话解释什么是 API"}
    ],
}

response = requests.post(API_URL, headers=headers, json=data, timeout=30)

print("状态码:", response.status_code)
print("返回内容:", response.text)

response.raise_for_status()
result = response.json()
print(result["choices"][0]["message"]["content"])