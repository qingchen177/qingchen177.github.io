---
title: Typora个人设置
layout: post
categories: [Wiki]
tags: [Typora, Mdtht]
---

记录一些个人的设置

### 主题

用的这个：https://github.com/cayxc/Mdtht（按照作者文档安装就行）

还有配套导出html的插件不要太好用！https://github.com/cayxc/Mdtht（按照作者文档安装就行）

世上还是好人多啊！

在 `偏好设置` -> `外观` -> `打开主题文件夹`把`mdmdt-light.css` 和 `mdmdt-dark.css`放进去，重启`typora`完成

然后微调一些个人喜好

#### 修改代码块样式

参考issue：https://github.com/cayxc/Mdmdt/issues/11

```css
.md-fences{
  padding-top: 32px;
  background: rgb(40, 42, 50) !important;
}

.md-fences::before {
  background: #fc625d;
  border-radius: 50%;
  box-shadow: 18px 0 #fdbc40, 36px 0 #35cd4b;
  content: ' ';
  height: 10px;
  left: 10px;
  margin-top: -22px;
  position: absolute;
  width: 10px;
}
.md-fences pre {
  color: rgb(187, 199, 253) !important;
}
```

#### 修改kbd样式

本来是黑色背景没有边框看着有点不习惯，这边改成这样

```css
kbd {
  display: inline-block;
  border: 1px solid;
  background: var(border-color);
  color: va(--bg-color2);
  border-image: none;
  border-radius: 5px;
  padding: 0 6px;
  font-size: 14px;
  box-shadow: none;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}
```
