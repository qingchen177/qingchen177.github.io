---
layout: post
title: Python代码片段001
categories: ["code"]
tags: ["python"]
---

## Python代码片段001

### 获取本机局域网IP地址

```python
# @Time    : 2025/8/14 18:52
def get_lan_ip():
    """
    获取本机局域网IP地址
    """
    import socket
    try:
        # 创建一个UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接一个外部地址（不需要真实连接）
        s.connect(("8.8.8.8", 80))
        # 获取本地IP
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


print(get_lan_ip())
```

### httpx.AsyncClient绕过代理

添加参数`proxy=None, trust_env=False`

```python
 async with httpx.AsyncClient(base_url=base_url, timeout=30, proxy=None, trust_env=False) as client:
    pass
```

