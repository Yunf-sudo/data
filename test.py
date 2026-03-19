import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

urls = {
    "Bing": "https://www.bing.com",
    "Houzz": "https://www.houzz.com",
    "Flickr": "https://www.flickr.com",
    "Dezeen": "https://www.dezeen.com"
}

print("开始测试四个网站的连通性...\n")

for name, url in urls.items():
    try:
        # 注意这里去掉了 proxies=proxies
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"✅ {name}: 访问成功 (状态码 200)")
        else:
            print(f"⚠️ {name}: 访问被拒绝或异常 (状态码 {response.status_code})")
    except Exception as e:
        print(f"❌ {name}: 连接完全失败 - 错误信息: {e}")