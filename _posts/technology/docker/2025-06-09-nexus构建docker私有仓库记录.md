---
layout: post
title: 通过Nexus构建一个Docker私有仓库记录
tags: [Docker,Nexus]
categories: [运维]
---

参考：https://blog.csdn.net/gengkui9897/article/details/127353727

> 这里就跟着博客一步步来的，简单讲一下步骤，主要记录搭建过程的问题

### Nexus Repository Manager

拉镜像
```bash
docker pull sonatype/nexus3:latest
# 或者指定版本
docker pull sonatype/nexus3:3.42.0
```
> [!NOTE]
>
> 20250609：这里拉最新的镜像碰到坑了，docker login会出问题，降版本后解决



```yaml
services:
  # nexus
  nexus:
    image: sonatype/nexus3:3.42.0
    volumes:
        - /$LJL_DOCKER/volume/nexus-data:/nexus-data #宿主机:容器
        - /etc/localtime:/etc/localtime:ro # 让容器的时钟与宿主机时钟同步，避免时间的问题，ro是read only的意思，就是只读。
    environment:
        - "TZ=Asia/Shanghai"
    ports:
      - "8081:8081" # web服务使用
      - "8082:8082" # http仓库使用
      - "8083:8083" # https仓库使用，本例不使用
    restart: always
    container_name: nexus
    networks:
      - ljl-network

networks:
  ljl-network:
    name: ljl-network
    driver: bridge
    external: true

```

这样就启动起来了

#### 配置

- 新建一个二进制存储（可选）
- 新建一个docker仓库（hosted）并配置
- 创建realms权限
- 创建docker role并赋权（这里在privileges特权这里，只给一个会导致无法推送拉取镜像）
- 添加用户

### Docker

#### 配置

修改`/etc/docker/daemon.json`

```json
{
    "registry-mirrors": [
        "https://dofaorfy.mirror.aliyuncs.com",
        "https://dockerproxy.com",
        "https://hub-mirror.c.163.com",
        "https://mirror.baidubce.com",
        "https://ccr.ccs.tencentyun.com",
        "https://docker-cf.registry.cyou",
        "https://dockercf.jsdelivr.fyi",
        "https://docker.jsdelivr.fyi",
        "https://dockertest.jsdelivr.fyi",
        "https://mirror.aliyuncs.com",
        "https://mirror.baidubce.com",
        "https://docker.m.daocloud.io",
        "https://docker.nju.edu.cn",
        "https://docker.mirrors.sjtug.sjtu.edu.cn",
        "https://docker.mirrors.ustc.edu.cn",
        "https://mirror.iscas.ac.cn",
        "https://docker.rainbond.cc",
        "https://docker.registry.cyou",
        "http://192.168.1.4:8082",
        "https://192.168.1.4:8083"
    ],
    "insecure-registries": [
        "192.168.1.4:8082",
        "192.168.1.4:8083"
    ],    
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         }	
    }
}
```

```bash
sudo sudo systemctl daemon-reload 
sudo systemctl restart docker
```

#### 登录

```shell
docker login -u cc -p 123456 192.168.1.4:8082
```

> [!NOTE]
>
> **我用最新版的会报错，这里nexus降级后没问题了**
>
> get "null://127.0.0.1:8082/v2/token?account=cc&client_id=docker&offline_token=true&service=null%3a%2f%2f127.0.0.1%3a8082%2fv2%2ftoken": unsupported protocol scheme "null"

#### 推送

```shell
docker tag brainy:latest  192.168.1.4:8082/brainy:1.0.0
docker push 192.168.1.4:8082/brainy:1.0.0
```

> [!note]
>
> 这里如果按照博客的特权只给一个的话，推送不了，全部加上后完成
>
> ![image-20250609164603054](/assets/images/post/image-20250609164603054.png)

#### 拉取

```shell
docker pull 192.168.1.4:8082/msxk-brainy:1.0.0
```

## END
