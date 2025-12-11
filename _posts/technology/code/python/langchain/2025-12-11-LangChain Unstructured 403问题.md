---
layout: post
title: LangChain Unstructured 403问题
tags: ["LangChain","LangServe","LangGraph"]
categories: ["Python"]
---

## 问题

打包 docker 镜像，运行时出现403错误

```python
loader = UnstructuredLoader(file_path)
docs = await loader.aload()
```

报错：
`HTTPError: HTTP Error 403: Forbidden`

血🐴坑
搞了大半天才找到问题原因。

可以参考这个 issue
[https://github.com/Unstructured-IO/unstructured/issues/3890](https://github.com/Unstructured-IO/unstructured/issues/3890)

## 解决
出现此问题的原因是默认的 NLTK_DATA_URL 已失效。建议使用 NLTK 原生方法直接下载所需的 NLTK 数据

添加代码

```python
import nltk
import os


def check_for_nltk_package(package_name: str, package_category: str) -> bool:
    """Checks to see if the specified NLTK package exists on the image."""
    paths: list[str] = []
    for path in nltk.data.path:
        if not path.endswith("nltk_data"):
            path = os.path.join(path, "nltk_data")
        paths.append(path)

    try:
        nltk.find(f"{package_category}/{package_name}", paths=paths)
        return True
    except (LookupError, OSError):
        return False


def download_nltk_packages():
    """If required NLTK packages are not available, download them."""

    tagger_available = check_for_nltk_package(
        package_category="taggers",
        package_name="averaged_perceptron_tagger_eng",
    )
    tokenizer_available = check_for_nltk_package(
        package_category="tokenizers", package_name="punkt_tab"
    )

    if (not tokenizer_available) or (not tagger_available):
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("punkt_tab", quiet=True)

```

启动的时候运行一下`download_nltk_packages()`
