---
layout: post
title: Jekyll在windows下安装的记录
tags: [Ruby,Jekyll,bundle,gem]
categories: [Wiki]
---

### ruby windows 安装
https://rubyinstaller.org/
下载安装，安装完成后点击确认后弹出黑框按enter最后等待安装完成 

```shell
ruby -v
gem -v

# 快速起一个jekyll网页
gem install bundler jekyll
jekyll new jekyll-demo
cd jekyll-demo
bundle exec jekyll serve
```
### gem install 缓慢
解决办法
1. 加代理
2. 改配置


修改配置命令

```bash
gem sources --remove https://rubygems.org/
gem sources -a https://gems.ruby-china.com/

gem install jekyll bundler

bundle config mirror.https://rubygems.org https://gems.ruby-china.com
```

### bundle exec jekyll serve报错
`cannot load such file -- csv (LoadError)`
解决：
```shell
bundle add csv #其他同理
```


{% raw %}

### jekyll
- `{% seo %}` 和 `{%- seo -%}` 都可以使用。
- 功能一样，都是为了生成 SEO 相关的 meta 标签。
- `{%- ... -%}` 这种写法更适合用于正式项目，可以避免不必要的空白输出。

{% endraw %}

#### 部署问题

20250924

本地`jekyll build`正常

但是在cloudflare上部署的时候，访问的时候报错

```shell
14:47:15.796	/opt/buildhome/.asdf/installs/ruby/3.4.4/lib/ruby/gems/3.4.0/gems/jekyll-4.4.1/lib/jekyll/url.rb:161:in 'String#encode': "\xE5" from ASCII-8BIT to UTF-8 (Encoding::UndefinedConversionError)
```

最终定位是因为图片文件名有中文

Cloudflare 的构建容器中，操作系统或文件系统将非 ASCII 文件名视为二进制数据（ASCII-8BIT/BINARY 编码），而 Ruby 尝试将其转换成 UTF-8 字符串时报错。

把图片名字都改成英文就行
