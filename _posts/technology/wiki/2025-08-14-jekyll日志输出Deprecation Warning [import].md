---
layout: post
title: jekyll日志输出Deprecation Warning...解决方案
tags: [Jekyll,Sass]
categories: [Wiki]
---

## jekyll日志输出Deprecation Warning [import]

用Mac装完环境启动jekyll会输出下面这种警告：

`Deprecation Warning [import]: Sass @import rules are deprecated and will be removed in Dart Sass 3.0.0.`

Jekyll版本 > 4.3，可以直接在`config.yml`中添加下面的配置：

```yaml
sass:
  quiet_deps: true          # 只静默来自依赖（node_modules、theme）的警告
  silence_deprecations:
    - import                # 完全屏蔽针对 @import 的弃用警告
    - global-builtin
```

