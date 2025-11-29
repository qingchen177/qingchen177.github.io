---
layout: post
title: 配置 Cloudflare 邮件转发&发送的记录
tags: ["Cloudflare", "Email"]
categories: ["工具"]
---

## 前情

目前邮箱主要用的是 gmail 和 qq，突然想用自己的域名搞一个邮箱玩一下。

网络搜到下面的非常好教程：

- [绅士喵 - 配置 Cloudflare 邮件转发](https://blog.hentioe.dev/posts/configure-cloudflare-email-forwarding.html)
- [绅士喵 - 使用 Gmail 作为 Cloudflare 域名邮箱的发信服务](https://blog.hentioe.dev/posts/send-emails-via-gmail-for-cloudflare-domain.html)

这两天整公司的企业邮箱，回头看我这个已经忘记是怎么弄的了，我感觉还是要记录一下。

> 又在制造垃圾了，但是又怕文章没了，自己也没备份，后面再搞那就是一头雾水。。。
>
> （虽然收藏了书签，但是书签太多了，google 书签已经乱了，经常找不到文章，最近书签分类整理后反而更难找了😓，加上现在已经习惯先来自己网站找记录了。）
>
> 这两篇文章作者按照 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hans) 授权，需要提供 `适当的署名`：
>
> <u>提供创作者和署名者的姓名或名称、版权标识、许可协议标识、免责标识和作品链接。</u>
>
> 老实说这些概念确实不太清楚，我把名称 - 连接都列出来了，有缺少的可以评论告知补充。

## 记录

当时跟着文章流程基本没啥问题，现在就是回顾简单的记录。

### 转发邮件

1. 进入  [Cloudflare](https://dash.cloudflare.com/) 
2. 进入域名的「[电子邮件](https://dash.cloudflare.com/?to=/:account/:zone/email)」页面
3. 进入「电子邮件路由」页面
4. 选择「目标地址」
5. 点击「添加目标地址」
6. 选择「路由规则」
7. 点击「创建地址」
8. 填写表单，选择目标地址
9. 即可（貌似要在「电子邮件路由 - 设置」中搞一下域名的 DNS，应该是一键自动就配好了，细节忘记了）
10. 子域名在「设置」中点击「添加子域」即可

### 发送邮件

1. Gmail 邮箱生成一个应用密码[此页面](https://security.google.com/settings/security/apppasswords)

2. [帐号和导入](https://mail.google.com/mail/u/0/#settings/accounts) 在「用这个地址发送邮件：」设置项的右边点击「添加其他电子邮件地址」。接着在弹出的新窗口中输入你的域名邮箱地址

3. 完成后点击「下一步」，输入 Gmail 的 SMTP 登录信息

4. 填入 smtp.gmail.com ，gmail以及生成的应用密码

5. 收到邮件然后确认邮件

6. DNS：更新SPF记录 `v=spf1 include:_spf.mx.cloudflare.net include:_spf.google.com ~all`

7. DNS：添加 DMARC 记录

   ```text
   类型：TXT
   名称：_dmarc
   TTL：自动
   内容："v=DMARC1; p=none; aspf=s; rua=mailto:<your-email-address>"
   ```

8. 测试发送：[www.mail-tester.com](https://www.mail-tester.com/) 应该得到 9 分

## END
