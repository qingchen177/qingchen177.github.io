---
layout: post
title: Mac Python 安装 Unstructured 相关依赖
categories: [操作系统]
tags: [Mac, Unstructured, LibreOffice, libmagic]
---

## 背景

Mac 安装 `unstructured` 时，碰见的两个问题。

## 命令

我用 `langchain` 的集成包装的

`langchain` 文档地址：http://docs.langchain.com/oss/python/integrations/document_loaders/unstructured_file#installation-for-local

`unstructured`官方网址：https://docs.unstructured.io/open-source/installation/full-installation

```shell
# base dependencies
brew install libmagic poppler tesseract

# If parsing xml / html documents:
brew install libxml2 libxslt

uv add "langchain-unstructured[local]"
```

## 问题

1. 找不到 `ligmagic` 的执行路径
2. 找不到 `soffice` 的执行路径

## 解决

### 问题 1 

```shell
uv add pylibmagic 
```

然后代码里面引入 `pylibmagic` 即可

### 问题 2

由于`LibreOffice`不会自动配置环境变量，需要手动配置一下。

```shell
# 安装LibreOffice
brew install --cask libreoffice
# 配置
sudo ln -s /Applications/LibreOffice.app/Contents/MacOS/soffice /usr/local/bin/soffice
```

