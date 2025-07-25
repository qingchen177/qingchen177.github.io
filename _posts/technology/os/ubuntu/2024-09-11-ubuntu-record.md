---
layout: post
title: Ubuntu记录
categories: [ 操作系统 ]
tags: [ Ubuntu,美化,环境安装 ]
#author: 清尘
#banner: "/assets/images/banners/3.jpg"
---
**前情提要**：算法岗，公司电脑配置3090显卡，装的是ubuntu，用了大概两个月装了很多东西，很爽，小毛病也很多，这两个月体验下来完全符合个人审美和体验，但在昨天跑模型的时候突然卡死，脑子一想当然是重启解决啦，重启后发现只有鼠标能动，应该是gnome桌面卡死，重启还是这样，于是用tty，发现也不行tty也用不了不弹出login，敲什么也没反应(我应该是装了什么其他的东西导致的，本人没用ubuntu的终端换成了之前一直用的，性格就是执着与一个东西舍不得换，真想改掉)，大概有2-3H都没什么解决办法，但是工作文件，模型，环境等等都在电脑里面，准备用启动盘进行修复，自己的启动盘在家里面，发现同事之前的启动盘还有（刚入职找他借得U盘装）本来是Ubuntu，结果同事哪天心血来潮换了win11，我当时有点没头目，就插上去看下能不能先保数据然后重装，到win11安装界面，看见硬盘点进去后，说无法加载，弹出来个按钮，当时也是脑子糊了，点了一下，点完弹出一下进度条立马清醒仔细回忆刚刚的英文是什么，狂点取消，但是为时已晚，仅仅1秒不到，空了（没错空了，我没打感叹号是因为我只能接受）。事以至此，干脆从头再来吧。

好，直接从头开始尽可能细的记录一下ubuntu和深度学习相关的东西吧，废话到此开始第一步！

### Ubuntu安装

#### 你需要有的

一台有网络用于AI学习的电脑（装ubuntu）、一个8G+的U盘（启动盘）、一台可以正常联网的电脑（下ubuntu系统镜像）

#### 镜像下载

清华镜像站：https://mirrors.tuna.tsinghua.edu.cn/#

![image-20241101153801016](/assets/images/post/image-20241101153801016-1730703095207-1.png)

选择后就会浏览器就会开始下载，然后等待的时间做启动盘

#### U盘制作

[参考这篇文章](/工具/2024/09/11/Ventoy.html)

#### 安装

U盘插到AI电脑开机然后什么F2、F12瞎jb按，进入BIOS设置u盘为第一启动后保存退出（大概这意思，没有图因为我走到这才想起来写这个）

然后就是这样的一个界面（官网偷的，我本来想手机拍，感觉太low了）

![img](/assets/images/post/screen_uefi_cn-1730703095207-6.png)

选择ubuntu然后会有四个选项

都选择第一个就行

`boot in normal`

`try or install ubuntu`

大致是这几个，可能有和我不一样的，选第一个基本没错。

然后就进入ubuntu界面开始傻瓜式安装，没图就口头描述一下：

（我是插网线的，无线网应该还会有一步联网）

1. 选择语言：`中文简体`
2. 点击`安装`
3. 点击`继续`（键盘默认会是chinese）【20241228：默认也有可能是英文，往上翻选Chinese就行】
4. 选择`正常安装`、其他选项选个`nvidia显卡驱动`，选完点击`继续`，然后就开始下驱动了等几分钟泡杯咖啡
5. 选择`清除磁盘安装ubuntu`
6. 选择`继续`，时区上海，然后`继续`
7. 设置密码，然后`继续`
8. 等一会安装完成重启即可（会提示你拔掉启动盘，然后按enter进入）

至此安装结束。

> 20241228：从Fedora换成Ubuntu记录
>
> 因为工作电脑是Ubuntu导致我现在自己的fedora老是打错安装命令，感觉有点分裂，看了笔记本里面没有什么东西，直接换Ubuntu算了
>
> 这里讲一下安装Ubuntu的分区设置：
>
> ```bash
> /dev/mapper/ventoy是启动盘设备（如果你的启动盘是按照我上面的去搞的）这个不用管
> /dev/sda 大小128G的固态硬盘
> /dev/sdb 大小1T是机械硬盘
> /dev/sdc 大小30G是U盘
> 我这边就是清空 固态和机械
> 把sda和sdb的下面的什么sda1、、、都删除变成空闲就行
> 然后我是这样分区的
> 固态：
> 200MB EFI系统分区
> 20GB 交换分区 （我这台老电脑就8g内存）
> 剩下的全部给根目录/
> 机械：
> 全给/home挂载，慢也没啥办法。。。
> 
> 注意！！！别点现在安装，
> 安装启动引导器的设备：
> 这里选择固态的EFI系统分区，不然白装
> 我这里是/dev/sda1选择然后安装
> ```

参考（我随便找的，能用就行）：https://blog.csdn.net/dengjin20104042056/article/details/130477959

## Ubuntu初始化

我这里就是记录我自己的探索过程：

重启完会弹出很多一个一个处理

1. `Ubuntu Pro skip`->`拒绝发送报告信息`->`允许位置信息`
2. `24.04 LTS 不升级`
3. `软件更新器直接关掉`

---

#### 系统-init

可以打开火狐浏览器访问CSDN官网：搜`ubuntu22.04 安装完后必做的事`

参考大佬的步骤

https://blog.csdn.net/ctu_sue/article/details/127051250

`ctrl+alt+t`打开终端

然后复制下面的

```shell
sudo apt-get remove --purge libreoffice*
sudo apt-get remove libreoffice-common
sudo apt-get remove unity-webapps-common
sudo apt-get remove thunderbird totem rhythmbox empathy brasero simple-scan gnome-mahjongg aisleriot onboard deja-dup
sudo apt-get remove gnome-mines cheese transmission-common gnome-orca webbrowser-app gnome-sudoku landscape-client-ui-install
sudo apt autoremove
sudo apt-get clean && sudo apt-get autoclean
```

终端`ctrl+shift+v`粘贴回车运行，中途需要输入三次`y`

![image-20241101185819192](/assets/images/post/image-20241101185819192.png)

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -f
sudo apt install fuse
# 我习惯用vim
sudo apt install vim -y
```

中途输入Y确认

![image-20241101191149219](/assets/images/post/image-20241101191149219.png)

***

#### 系统-软件源

打开软件和更新，`下载自`在其他里面选阿里的

![image-20241101203721853](/assets/images/post/image-20241101203721853.png)

---

#### 系统-python

然后软连接一下python

1. 直接输入`python`没有
2. 输入`python3 –version`显示python3的版本
3. `sudo ln -s /usr/bin/python3 /usr/bin/python`
4. `python`有结构成功，`ctrl+d`退出



![image-20241101191251300](/assets/images/post/image-20241101191251300.png)

***

#### 系统-个人文件夹

因为个人目录下的文件夹是中文终端不方便，改成英文，打开设置（不是哥们怎么设置打不开了，应该是上面卸载什么一起卸载掉了）

##### Emergency-设置打不开

重新安装执行如下命令

```shell
sudo apt-get install unity-control-center
sudo apt-get install gnome-control-center
```

然后注销用户在进入就有设置了，选择英语

![image-20241101192706660](/assets/images/post/image-20241101192706660.png)

然后重启进来这个页面选择`Update Names`

![image-20241101192911607](/assets/images/post/image-20241101192911607.png)

然后再切回中文

![image-20241101193120986](/assets/images/post/image-20241101193120986.png)

---

#### 系统-SSH

```shell
sudo apt install openssh-server -y
sudo systemctl enable ssh
```

----

#### 软件-向日葵

先安装这个是因为这样可以下班回家远程公司电脑装系统，为了不掉进度，并不是因为我爱上这个B班

https://sunlogin.oray.com/download/linux?type=personal&ici=sunlogin_navigation

![image-20241101194106930](/assets/images/post/image-20241101194106930.png)

下载完成后，打开文件夹，打开终端输入（记得替换成你的软件包，先输入`sudo dpkg -i S`然后按 tab可以补全）

`sudo dpkg -i SunloginClient_15.2.0.63064_amd64.deb`

不出意外出意外了，有错误

执行命令：

`sudo apt-get install -f -y`

![image-20241101194909821](/assets/images/post/image-20241101194909821.png)

完成向日葵安装

![image-20241101194948566](/assets/images/post/image-20241101194948566.png)

如果你还有问题参考向日葵官网解决https://service.oray.com/question/8286.html

---

#### 环境-nvidia

先试探三连

```bash
nvidia-smi
nvcc -V
ls /usr/local/cuda
```

![image-20241101200506087](/assets/images/post/image-20241101200506087.png)

然后看下`/usr/local`下面有没有cuda-xxx的目录，可能你装了一个版本的

![image-20241101200743434](/assets/images/post/image-20241101200743434.png)

我这表明我装了显卡驱动但是没有装cuda

ok开始装CUDA

看nvidia-smi的CUDA Version：12.4

表明最高支持CUDA12.4

那就装12.4

https://developer.nvidia.com/cuda-toolkit-archive

![image-20241101201056038](/assets/images/post/image-20241101201056038.png)

12.4.1点击进入，选择`Linux`-`x86_64` -`Ubuntu` -`22.04` -`runfile(local)`

![image-20241101201400474](/assets/images/post/image-20241101201400474.png)

然后就会出现命令复制自己运行：

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

然后进入安装界面

![image-20241101201933453](/assets/images/post/image-20241101201933453.png)

选择`Continue`报错没有gcc

执行`sudo apt install gcc dkms`安装

![image-20241101202300716](/assets/images/post/image-20241101202300716.png)

然后再执行`sudo sh cuda_12.4.1_550.54.15_linux.run`

选择`Continue`

输入`accept`

![image-20241101202658411](/assets/images/post/image-20241101202658411.png)

选择如下，然后安装，截图错了，别选`nvidia-fs`

![image-20241101202901337](/assets/images/post/image-20241101202901337.png)

出现如下安装完成：

![image-20241101204630438](/assets/images/post/image-20241101204630438.png)

然后配置环境变量

```shell
sudo gedit ~/.bashrc
```

把下面的粘贴进去

```bash
export PATH=$PATH:/usr/local/cuda/bin  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64  
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
```

```bash
source ~/.bashrc
```

验证

`nvcc -V`

![image-20241101205222101](/assets/images/post/image-20241101205222101.png)

安装CUDNN

网址：https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

```shell
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

一路顺利完成安装

![image-20241101210727538](/assets/images/post/image-20241101210727538.png)

---

#### 环境-Anaconda安装

https://www.anaconda.com/download/success

下载

![image-20241101204447712](/assets/images/post/image-20241101204447712.png)

运行

```bash
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh 
```

按`Enter`然后一直按`Enter`然后出现yes，输入`yes`然后`enter`然后`yes`，出现下面界面完成安装yes

![image-20241101210051575](/assets/images/post/image-20241101210051575.png)

然后配置一下清华源

参考网址：https://mirror.tuna.tsinghua.edu.cn/help/anaconda/

`sudo gedit ~/.condarc`

替换成如下的：

```shell
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

使用下列命令清除索引缓存，并安装常用包测试一下。

```shell
conda clean -i
conda create -n test numpy
```

最后打开新终端

输入`conda config --set auto_activate_base false`

设置默认不加载base环境

![image-20241101210230141](/assets/images/post/image-20241101210230141.png)

![image-20241101210348012](/assets/images/post/image-20241101210348012.png)

conda安装完成

---

#### 环境-pytorch安装

https://pytorch.org/

```shell
conda create --name pytorch
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# 耗时有点长这块不知道加代理会不会快点，反正我就等着了很呆。。。好像是已经开了梯子的不知道为啥很慢
```

![image-20241101230015521](/assets/images/post/image-20241101230015521.png)

```shell
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
```

检测环境是否安装成功

PS：

```shell
# 临时清华源
# some-package代表你需要安装的包
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

#### 环境-Docker

https://www.docker.com/

https://developer.aliyun.com/article/110806

```bash
sudo apt-get autoremove docker docker-ce docker-engine docker.io containerd runc
sudo dpkg -l | grep docker
# 没有显示表示没装继续
```

```bash
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker --version
#docker设置开机启动
sudo systemctl enable docker.service
```

![image-20241101232238260](/assets/images/post/image-20241101232238260.png)

设置非root用户也能操作docker

```shell
# 把li换成你的用户名
sudo usermod -aG docker li
```

~~然后注销用户在进来~~

重启就可以了

![image-20241101234859979](/assets/images/post/image-20241101234859979.png)

配置阿里镜像加速

https://cr.console.aliyun.com/cn-beijing/instances/mirrors

```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
      "https://wb640ddh.mirror.aliyuncs.com",# 换成你的
      "https://dockerproxy.com",
      "https://hub-mirror.c.163.com",
      "https://mirror.baidubce.com",
      "https://ccr.ccs.tencentyun.com"
  ] 
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

```json
# 20241112更新：daemon.json内容如下，不知道为什么拉不下来镜像，改成下面的就可以了
# 20241206更新：加上NVIDIA
# 20250714更新：好多都不能地址用了，能用梯子用梯子吧，不过还是很耗流量的，自己用nexus搭一个docker镜像仓库好一点
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
        "https://docker.registry.cyou"
    ],
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         }	
    }
}
```

##### docker 清理

```shell
docker system prune -a -f --volumes
```

##### 安装Installing the NVIDIA Container Toolkit

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl daemon-reload
sudo systemctl restart docker
```

##### Docker-WARNING: bridge-nf-call-iptables is disabled

`docker info`最后出现两行

```bash
WARNING: bridge-nf-call-iptables is disabled
WARNING: bridge-nf-call-ip6tables is disabled
```

解决：

```shell
vim /etc/sysctl.conf
# 在最后加上
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
sysctl -p
systemctl restart docker
```
---

#### 系统-sudo免密码

```shell
sudo su  				#切换为root用户
chmod u+w /etc/sudoers
vi /etc/sudoers   #如果不会使用vim编辑器可以用其他文本编辑器，能修改文件就行
```

```shell
# 找到这一行
# Allow members of group sudo to execute any command
%sudo	ALL=(ALL:ALL) ALL
#下面的user是你的用户名 添加这一行就能免密sudo了
user ALL=(ALL:ALL) NOPASSWD: ALL
```

```shell
chmod u-w /etc/sudoers   #别忘了去掉sudoers文件的写入权限
```

![image-20241101232734528](/assets/images/post/image-20241101232734528.png)

![image-20241101232810610](/assets/images/post/image-20241101232810610.png)

测试成功

![image-20241101233038619](/assets/images/post/image-20241101233038619.png)

然后改一下root的终端命令行展示

```bash
echo $PS1
# 复制结果
\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$
sudo su
vim /root/.bashrc
# 在最后一行加
PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$'
# :wq 保存退出
source /root/.bashrc
```

![image-20241101233533827](/assets/images/post/image-20241101233533827.png)

---


#### Docker-v2raya

20241228：不可控因素，上面docker的镜像源都失效了，emm这就出现一个问题，你想看world，首先你得在world里面，哈哈

```shell
docker pull mzz2017/v2raya
docker container stop v2raya
docker container rm v2raya
docker run -d \
  --restart=always \
  --privileged \
  --network=host \
  --name v2raya \
  -e V2RAYA_LOG_FILE=/tmp/v2raya.log \
  -e V2RAYA_V2RAY_BIN=/usr/local/bin/v2ray \
  -e V2RAYA_NFTABLES_SUPPORT=off \
  -e IPTABLES_MODE=legacy \
  -v /lib/modules:/lib/modules:ro \
  -v /etc/resolv.conf:/etc/resolv.conf \
  -v /etc/v2raya:/etc/v2raya \
  mzz2017/v2raya
docker container stats v2raya

# 访问 http://127.0.0.1:2017/
# END
```

> [!TIP]
>
> 20250311：我的订阅在V2raya不能选择指定的ip访问chatgpt，我现在换成了clash
>
> ```shell
> # https://github.com/clash-verge-rev/clash-verge-rev
> # release 下载deb包安装即可
> sudo dpkg -i Clash.Verge_2.3.1_amd64.deb
> sudo apt --fix-broken install
> ```

---

#### 软件-谷歌浏览器

```shell
# https://www.google.cn/chrome/next-steps.html?statcb=0&installdataindex=empty&defaultbrowser=0
 sudo dpkg -i google-chrome-stable_current_amd64.deb 
```

---

#### 软件-snap更新、Flathub安装

```shell
sudo snap refresh snap-store
# kill 进程
sudo snap refresh snap-store
```

![image-20241103001134428](/assets/images/post/image-20241103001134428.png)

```shell
sudo apt install flatpak -y
flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
# 重启电脑
sudo flatpak remote-modify flathub --url=https://mirror.sjtu.edu.cn/flathub
```

---

#### 软件-qq、微信、飞书

20241122更新：Linux微信有测试版！！！

https://linux.weixin.qq.com/

卸载flatpak的微信：

```shell
flatpak list
flatpak uninstall com.tencent.WeChat
flatpak uninstall --unused
sudo dpkg -i WeChatLinux_x86_64.deb
```

之前的：

```shell
# qq https://im.qq.com/linuxqq/index.shtml
sudo dpkg -i QQ_3.2.13_241023_amd64_01.deb 

# wechat 有点小bug但还能用参考https://github.com/dushaoshuai/dushaoshuai.github.io/issues/148
flatpak install flathub com.tencent.WeChat
flatpak override --user --filesystem=home com.tencent.WeChat
# 然后top bar打不开的问题装插件AppIndicator and KStatusNotifierItem Support解决

# https://www.atzlinux.com/allpackages.htm
# 或者 但是感觉不好用，之前找到好像是一个官方的，但现在找不到了，后面找到了再更新
wget -c -O atzlinux-v12-archive-keyring_lastest_all.deb https://www.atzlinux.com/atzlinux/pool/main/a/atzlinux-archive-keyring/atzlinux-v12-archive-keyring_lastest_all.deb
chmod 777 atzlinux-v12-archive-keyring_lastest_all.deb 
sudo apt -y install ./atzlinux-v12-archive-keyring_lastest_all.deb
sudo apt update
sudo cp /etc/lsb-release /etc/lsb-release.Ubuntu
sudo apt -y install electronic-wechat-icons-atzlinux
sudo apt -y install com.tencent.wechat
sudo cp /etc/lsb-release /etc/lsb-release.wechat

# 飞书 https://www.feishu.cn/download
 sudo dpkg -i Feishu-linux_x64-7.22.9.deb 
```

微信打不开参考在优化的时候装的插件[Ubuntu界面](#h-ubuntu界面)

---

#### 警告-密钥存储在过时的 trusted.gpg 

密钥存储在过时的 trusted.gpg 密钥环中（/etc/apt/trusted.gpg），请参见 apt-key(8) 的 DEPRECATION 一节以了解详情

```shell
cd /etc/apt
sudo cp trusted.gpg trusted.gpg.d
```

---

#### 系统-java-1.8

```shell
sudo apt-get install openjdk-8-jdk
java -version
```

---

#### 系统-git 、neofetch、git lfs

```shell
sudo apt-get install git neofetch
```

##### neofetch

![image-20241104142025245](/assets/images/post/image-20241104142025245.png)

##### git配置

```shell
cd ~/.ssh/
ll
# 中间Enter按过去
ssh-keygen -t rsa -C "qingchen0607@qq.com"
# 复制贴到github ssh key里面
cat id_rsa.pub
```

![image-20241104140644216](/assets/images/post/image-20241104140644216.png)

测试报错（不知道是不是v2raya的原因）：

![image-20241104141133971](/assets/images/post/image-20241104141133971.png)

解决：

```bash
ssh -T -p 443 git@ssh.github.com
vim ~/.ssh/config
# 加上
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
# 然后测试ok
ssh -T git@github.com
```

![image-20241104141635338](/assets/images/post/image-20241104141635338.png)

![image-20241104141755762](/assets/images/post/image-20241104141755762.png)

##### git lfs

https://git-lfs.com/

下载安装

使用

```shell
git lfs install
git lfs track "data/large-file.zip"
git add .gitattributes
git add data/large-file.zip
git commit -m "Add large file using Git LFS"
git push origin <branch-name>
```

---

#### 开发-jetbrain toolbox

vscode会快一点，但我一直用的jetbrain。

https://www.jetbrains.com.cn/toolbox-app/

`jetbrains-toolbox-2.5.1.34629.tar.gz`

解压然后双击即可运行

![image-20241103021412559](/assets/images/post/image-20241103021412559.png)

然后pycharm和idea安装一下

破解（em后面得买一个支持一下，现在是真穷）：

https://blog.csdn.net/qq_39599719/article/details/140148532?spm=1001.2014.3001.5502

https://blog.csdn.net/qq_39599719/article/details/140122617?spm=1001.2014.3001.5502

按步骤就行，调整vmoption是顺带调一下最大内存给多点。（20241104：新版本好像破解有问题，降低版本是可以的，jetbrain toolbox也有vmoption最好也加一下）

---

#### 软件-搜狗输入法

跟着官方文档做（往下翻按照这个做**Ubuntu20.04安装搜狗输入法步骤**）

https://shurufa.sogou.com/linux/guide

`20250607：目前用下来没有网上说的乱七八糟的问题，就是不能登录自己的账号，智能的感觉少了，打个长句没有win和手机上智能调整`

```shell
sudo apt-get install fcitx
sudo dpkg -i sogoupinyin_4.2.1.145_amd64.deb
sudo cp /usr/share/applications/fcitx.desktop /etc/xdg/autostart/
sudo apt purge ibus
sudo apt install libqt5qml5 libqt5quick5 libqt5quickwidgets5 qml-module-qtquick2
sudo apt install libgsettings-qt1
# 重启后安装完成
```

---

#### 软件-小合集

##### wps

https://www.wps.cn/product/wpslinux

虽然我更喜欢office，但是linux自带的真的用不惯，WPS还给了deb和rpm包，我真的哭死，也算是好感上来了一点，还好linux wps没那么难用

```shell
 sudo dpkg -i wps-office_12.1.0.17885_amd64.deb 
 # 然后处理一下字体小问题
 # Linux 系统缺失字体：Symbol､Wingdings､Wingdings 2､Wingdings 3､Webdings､MT Extra,WPS无法正确地显示某些符号(公式)！
 # 参考：https://blog.csdn.net/u013669912/article/details/139678684
 git clone https://github.com/winunix/wps-office-fonts.git
 cd wps-office-fonts
 sudo cp * /usr/share/fonts/
 sudo mkfontscale
 sudo fc-cache
```

![image-20241104134912354](/assets/images/post/image-20241104134912354.png)

##### 7z

```
https://sparanoid.com/lab/7z/download.html
# 下载解压就可以用了
```

##### snipaste

https://www.snipaste.com/download.html

```
# 下载后Snipaste-2.10.2-x86_64.AppImage属性设置为可执行文件
```

##### Typora

https://blog.csdn.net/treasure0911/article/details/136638826

##### WindTerm

Windows之前经常用的，linux也拿过来用当xshell，主要看中了功能：

- 自动复制选中文本
- 右键粘贴很方便

默认是普通，要自己设置的（不会就搜一下，都是中文摸索一下就行）

> 发现个问题就是在windterm里面复制的，不能拿出来在外面用，有点难顶，但是可以接受，之前还没有这个情况

https://github.com/kingToolbox/WindTerm/releases

下载对应版本的包

![image-20241104105935970](/assets/images/post/image-20241104105935970.png)

> 2.6.1版本有bug，我换成了2.4.1

解压就可以用了

然后设置一下快捷方式，安装文件夹自带一个desktop文件，拿过来改一下

![image-20241104111343168](/assets/images/post/image-20241104111343168.png)

```shell
[Desktop Entry]
Name=WindTerm
Comment=A professional cross-platform SSH/Sftp/Shell/Telnet/Serial terminal
GenericName=Connect Client
Exec=/home/li/software/WindTerm_2.4.1/WindTerm # 改成你的路径
Type=Application
Icon=/home/li/software/WindTerm_2.4.1/windterm.png # 改成你的路径
StartupNotify=false
StartupWMClass=Code
Categories=Application;Development
Actions=new-empty-window
Keywords=windterm
```

然后复制过去，这时候如果你写的正确就会出现图标，右键添加收藏夹即可

```shell
sudo cp windterm.desktop /usr/share/applications/windterm.desktop
sudo chmod -x windterm.desktop 
```

然后创建快捷键和终端一样快速打开：

`设置-键盘-自定义`

![image-20241104114900502](/assets/images/post/image-20241104114900502.png)

自己设置个快捷键就行

![image-20241104115017478](/assets/images/post/image-20241104115017478.png)

顺便挂个软连接终端也好快速打开（基本用不上但是加一下偶尔用）

```shell 
sudo ln -s /home/li/software/WindTerm_2.4.1/WindTerm /usr/bin/windterm
```

![image-20241104114717521](/assets/images/post/image-20241104114717521.png)

##### notepad–/notepadqq

或者

```bash
sudo apt install notepadqq
```

https://gitee.com/cxasm/notepad--

```bash
git clone https://github.com/cxasm/notepad--.git
cd notepad--
sudo apt-get install g++ make cmake -y
sudo apt-get install qtbase5-dev qt5-qmake qtbase5-dev-tools libqt5printsupport5 libqt5xmlpatterns5-dev -y
cmake -B build -DCMAKE_BUILD_TYPE=Release
cd build && make -j
cpack
```

如果顺利就是我这样出现deb包：

![image-20241104140020233](/assets/images/post/image-20241104140020233.png)

安装

```shell
sudo dpkg -i notepad--_1.22.0_amd64.deb
```

##### xmind

```bash
sudo dpkg -i Xmind-for-Linux-amd64bit-24.09.13001-202409190153.deb 
```

##### qqmusic

安装完会闪退

```bash
vim /usr/share/applications/qqmusic.desktop 
# 加上--no-sandbox，完整如下：
[Desktop Entry]
Name=qqmusic
Exec=/opt/qqmusic/qqmusic --no-sandbox %U
Terminal=false
Type=Application
Icon=qqmusic
StartupWMClass=qqmusic
Comment=Tencent QQMusic
Categories=AudioVideo;
```

然后重启电脑就好了

#### 软件-虚拟机

##### Virtual Box

https://www.virtualbox.org/
老牌选手，我现在用的少了，vmware个人免费了

#####  VMware Workstation Pro

参考：https://www.cnblogs.com/EthanS/p/18211302

1. 访问 [Broadcom 注册页面](https://profile.broadcom.com/web/registration)，通过邮箱注册或登录你的账户。

2. 依次选择「Software」>「VMware Cloud Foundation」>「My Downloads」。

3. 下载文件（我的是这样：`VMware-Workstation-Full-17.6.2-24409262.x86_64.bundle`）

4. ```shell
   sudo chmod +x VMware-Workstation-Full-17.6.2-24409262.x86_64.bundle
   sudo ./VMware-Workstation-Full-17.6.2-24409262.x86_64.bundle 
   ```

   ```shell
   li@li:~/Downloads$ sudo chmod +x VMware-Workstation-Full-17.6.2-24409262.x86_64.bundle 
   li@li:~/Downloads$ ./VMware-Workstation-Full-17.6.2-24409262.x86_64.bundle 
   Extracting VMware Installer...done.
   root access is required for the operations you have chosen.
   li@li:~/Downloads$ sudo ./VMware-Workstation-Full-17.6.2-24409262.x86_64.bundle 
   Extracting VMware Installer...done.
   Installing VMware Workstation 17.6.2
       Configuring...
   [######################################################################] 100%
   Installation was successful.
   ```

安装完成~

---

#### 硬件-移动硬盘挂载

ubuntu移动硬盘自动挂载报错：Error mounting: wrong fs type, bad option, bad superblock on /dev/sda1问题

```shell
sudo dmesg |tail
sudo apt install ntfs-3g
sudo ntfsfix -d /dev/sda1
```



硬盘相关命令：

```shell
fdisk -l
lsblk  //查看磁盘大小
df -h
du -sh   //查看当前目录大小
```



---

## Ubuntu界面

前言：就是瞎折腾，找了个B站教程跟着做，这是记录

【【Ubuntu美化】Ubuntu20.04桌面美化全过程】 https://www.bilibili.com/video/BV1Q3411c7ru/?share_source=copy_web&vd_source=56e805ecfcbd1e50e2f1fba350663947

美化文件：https://wwi.lanzoup.com/iOH3Q06v525a

```shell
sudo apt install gnome-tweaks chrome-gnome-shell gnome-shell-extensions -y

sudo apt install gtk2-engines-murrine gtk2-engines-pixbuf  -y

sudo apt install sassc optipng inkscape libcanberra-gtk-module libglib2.0-dev libxml2-utils -y
```

然后就有优化工具了，自己设置

https://extensions.gnome.org/

```shell
# 扩展工具
User Themes # 主题
Coverflow Alt-Tab # 20241103更新这玩意不喜欢不用了
OpenWeather # 天气
NetSpeed # 网速
Blur my shell # 透明
dash to dock # dock设置
Hide Top Bar # 隐藏顶栏
AppIndicator and KStatusNotifierItem Support # 解决微信top bar打不开的问题
# 农历日历安装
# 先下载这个https://gitlab.gnome.org/Nei/ChineseCalendar/-/archive/20240107/ChineseCalendar-20240107.tar.gz 
# 解压然后 ./install.sh (在当前用户下安装)
# 然后去安装这个就行了
# https://extensions.gnome.org/extension/675/lunar-calendar/ 
```

诶呀自己调吧，这个懒得记录了，随时会变就记一下插件吧。。。

`AppIndicator and KStatusNotifierItem Support`安装完记得关掉系统自带的。

(20241229：美化后要注销，然后在登录界面最右下角有个齿轮设置，选择`ubuntu-xorg`(应该是叫这个)，不然会出现top bar和dash在全屏应用下闪烁，snipaste也不能用，换了就都正常了)

![image-20241104104440139](/assets/images/post/image-20241104104440139.png)

最终效果：

![image-20241104161549401](/assets/images/post/image-20241104161549401.png)

---

## Ubuntu工具/脚本/命令

#### 工具-OCR文字识别

[参考](https://devpress.csdn.net/linux/66cfe431a1ed2f4c853e94f8.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MzE5NDU3LCJleHAiOjE3Mjk2NDgyNzMsImlhdCI6MTcyOTA0MzQ3MywidXNlcm5hbWUiOiJxcV80MDg5Mzk0MiJ9.VuLkeklZhBVb6FzMxeferkmoUCx6lFQzKstT5vVp7_M&spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-4-124059358-blog-111116693.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-4-124059358-blog-111116693.235%5Ev43%5Epc_blog_bottom_relevance_base9&utm_relevant_index=9)

##### 安装tesseract

```shell
# 添加源
sudo add-apt-repository ppa:alex-p/tesseract-ocr
# 更新源 
sudo apt update 
# 安装
sudo apt install tesseract-ocr 
```

##### 安装字库

tesseract支持60多种语言的识别，使用之前需要先下载对应语言的字库；
完整字库下载地址：
[https://github.com/tesseract-ocr/tessdata](https://link.csdn.net/?target=https%3A%2F%2Fgithub.com%2Ftesseract-ocr%2Ftessdata%3Flogin%3Dfrom_csdn)
简中英字库下载地址：
[https://share.weiyun.com/5IJtlcY](https://link.csdn.net/?target=https%3A%2F%2Fshare.weiyun.com%2F5IJtlcY%3Flogin%3Dfrom_csdn)

下载完成之后把`.traineddata`后缀名字库文件放到tessdata目录下，默认路径是`/usr/share/tesseract-ocr/4.00/tessdata`

```bash
sudo cp *.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
# 安装gnome-screenshot，xclip，imagemagick
sudo apt install gnome-screenshot xclip imagemagick -y
```

##### shell

先新建文件夹放临时截图文件

```shell
#我的在这
/home/li/Pictures/ocr
```

新建脚本`ocr.sh`

![image-20241104154318366](/assets/images/post/image-20241104154318366.png)

编辑脚本

```shell
#!/bin/env zsh 
SCR="/home/li/Pictures/ocr/temp"
SCR2="/home/li/Pictures/ocr/temp2"
# take a shot what you wana to OCR to text
gnome-screenshot -a -f $SCR.png

# increase the png
mogrify -modulate 100,0 -resize 400% $SCR.png 
# should increase detection rate

# OCR by tesseract
tesseract $SCR.png $SCR &> /dev/null -l eng+chi1

# get the text and copy to clipboard

#sed -i 's/[[:space:]]//g' $SCR.txt # 删除空格方式1
#sed -i 's/\ //g' $SCR.txt  # 删除空格方式2
cat $SCR.txt  | sed -r 's/([^0-9a-z])?\s+([^0-9a-z])/\1\2/ig'>$SCR2.txt  # 解决每个汉字之间有空格的情况，英文单词间空格依旧保留
cat $SCR2.txt | xclip -selection clipboard
exit
```

```shell
# 赋权
sudo chmod a+x ocr.sh
```

设置快捷键

`bash /home/li/Documents/work/scripts/ocr/och.sh`

![image-20241104155755379](/assets/images/post/image-20241104155755379.png)

测试：

`按F2 -> 出现截屏 -> 选中文字 -> 粘贴`

>  ocr效果一般般但是平时方便快速转文字

#### 脚本-nohup

老是忘记怎么写的记录一下

```shell
nohup /path/to/your/script.sh > /path/to/your/logfile.log 2>&1 &
```

#### 脚本-定时备份

定时任务

[参考](https://blog.csdn.net/qq_40243750/article/details/140169285)

`backup.sh`内容如下：

```
rsync -avz /home/li/Documents/work/ /media/li/qingchendisk/company/work/
rsync -avz /home/li/work/projects/ /media/li/qingchendisk/company/projects/
rsync -avz /home/li/work/scripts/ /media/li/qingchendisk/company/scripts/
```

打开终端

输入`crontab -e`（我选择2，我喜欢用vim，因为我只会用vim）

```shell
30 12 * * * /bin/bash /home/li/work/scripts/backup.sh
30 18 * * * /bin/bash /home/li/work/scripts/backup.sh
0 2 * * * /bin/bash /home/li/work/scripts/backup.sh
```

相关解释：

```bash
30 12 * * * /path/to/your/command
30 18 * * * /path/to/your/command
0 2 * * * /path/to/your/command
这里的/path/to/your/command是你想要执行的命令或脚本的路径。每个字段的含义如下：

第一个字段（分钟）：30表示在小时的第30分钟。
第二个字段（小时）：12表示中午12点，18表示晚上6点，2表示凌晨2点。
第三个字段（日）：*表示每天。
第四个字段（月）：*表示每个月。
第五个字段（星期）：*表示每天，不指定星期。
```

#### 工具-Rainlendar

http://www.rainlendar.net/

[Ubuntu桌面日历工具Rainlendar及便签工具indicator stickynotes](https://www.cnblogs.com/pipci/p/15981148.html)

#### 工具-便签indicator-stickynotes 

```bash
sudo add-apt-repository ppa:umang/indicator-stickynotes
sudo apt-get update 
sudo apt-get install indicator-stickynotes 
```

#### 工具-视频剪辑openshot

https://www.openshot.org/user-guide/

下载AppImage然后设置为可执行文件即可

#### 脚本-expect

```bash
#!/usr/bin/expect
#  加上面这个就是隐式调用
# 进入工作目录
cd /home/ubuntu/workspace/project
# 执行 git pull
spawn git pull origin master
expect "Username for 'http://ip:port':"
send "name\n"
expect "Password for 'http://name@ip:port':"
send "password\n"
expect eof
```

也可以显示调用

```bash
expect  上面的脚步的名字.sh
```

#### 工具-gimp图片编辑软件

地址：https://www.gimp.org/

#### 工具-WireShark抓包

```shell
sudo apt install wireshark
# 中途选择yes
sudo dpkg-reconfigure wireshark-common
# 中途选择yes
sudo usermod -aG wireshark $USER
newgrp wireshark
sudo reboot
```

#### 命令-查看磁盘空间

##### df

```shell
root@li:/# df -h
文件系统        大小  已用  可用 已用% 挂载点
tmpfs           6.3G  2.9M  6.3G    1% /run
/dev/nvme0n1p3  140G   49G   85G   37% /
tmpfs            32G  426M   31G    2% /dev/shm
tmpfs           5.0M  4.0K  5.0M    1% /run/lock
efivarfs        192K  155K   33K   83% /sys/firmware/efi/efivars
/dev/nvme0n1p1  476M  6.1M  469M    2% /boot/efi
/dev/nvme0n1p4  756G  673G   45G   94% /home
tmpfs           6.3G  196K  6.3G    1% /run/user/1000
overlay         140G   49G   85G   37% /var/lib/docker/overlay2/59e33b1e6f3f634ad3306a6d6706fda66cb23574b7728a64bab38a2477f6b518/merged
overlay         140G   49G   85G   37% /var/lib/docker/overlay2/9666de73dd5066ffc8a6169bf738389e87f0b4398d5457740cd452aa954fc7d4/merged
overlay         140G   49G   85G   37% /var/lib/docker/overlay2/33a5ff082f72dc520c10819d632cc003b3104017bb8157a49bc094bcc18ebf9c/merged
```

##### lsblk

```shell
root@li:/# lsblk
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0         7:0    0     4K  1 loop /snap/bare/5
loop1         7:1    0  73.9M  1 loop /snap/core22/1908
loop2         7:2    0  73.9M  1 loop /snap/core22/1963
loop3         7:3    0 505.1M  1 loop /snap/gnome-42-2204/176
loop4         7:4    0   516M  1 loop /snap/gnome-42-2204/202
loop5         7:5    0  91.7M  1 loop /snap/gtk-common-themes/1535
loop6         7:6    0  12.9M  1 loop /snap/snap-store/1113
loop7         7:7    0  12.2M  1 loop /snap/snap-store/1216
loop8         7:8    0  44.4M  1 loop /snap/snapd/23771
loop9         7:9    0  50.9M  1 loop /snap/snapd/24505
loop10        7:10   0   500K  1 loop /snap/snapd-desktop-integration/178
loop11        7:11   0   568K  1 loop /snap/snapd-desktop-integration/253
sda           8:0    0   1.8T  0 disk 
└─sda1        8:1    0   1.8T  0 part 
nvme0n1     259:0    0 931.5G  0 disk 
├─nvme0n1p1 259:1    0   476M  0 part /boot/efi
├─nvme0n1p2 259:2    0  19.1G  0 part [SWAP]
├─nvme0n1p3 259:3    0 143.1G  0 part /
└─nvme0n1p4 259:4    0 768.9G  0 part /home
```

```shell
root@li:/# sudo blkid /dev/sda1
/dev/sda1: LABEL="qingchendisk" BLOCK_SIZE="512" UUID="E834AA0634A9D838" TYPE="ntfs" PARTUUID="e65c76fd-01"
```

##### fdisk

```shell
root@li:/# fdisk -l
Disk /dev/loop0：4 KiB，4096 字节，8 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop1：73.89 MiB，77475840 字节，151320 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop2：73.89 MiB，77479936 字节，151328 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop3：505.09 MiB，529625088 字节，1034424 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop4：516.01 MiB，541073408 字节，1056784 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop5：91.69 MiB，96141312 字节，187776 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop6：12.93 MiB，13553664 字节，26472 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop7：12.2 MiB，12791808 字节，24984 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/nvme0n1：931.51 GiB，1000204886016 字节，1953525168 个扇区
Disk model: KINGSTON SNV2S1000G                     
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节
磁盘标签类型：gpt
磁盘标识符：2434E4AD-DC07-461D-94E6-3D21BBDA6077

设备                起点       末尾       扇区   大小 类型
/dev/nvme0n1p1      2048     976895     974848   476M EFI 系统
/dev/nvme0n1p2    976896   40976383   39999488  19.1G Linux swap
/dev/nvme0n1p3  40976384  340975615  299999232 143.1G Linux 文件系统
/dev/nvme0n1p4 340975616 1953523711 1612548096 768.9G Linux 文件系统


Disk /dev/loop8：44.45 MiB，46604288 字节，91024 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop9：50.89 MiB，53366784 字节，104232 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop10：500 KiB，512000 字节，1000 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/loop11：568 KiB，581632 字节，1136 个扇区
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节


Disk /dev/sda：1.82 TiB，2000398931968 字节，3907029164 个扇区
Disk model: EXTERNAL_USB    
单元：扇区 / 1 * 512 = 512 字节
扇区大小(逻辑/物理)：512 字节 / 512 字节
I/O 大小(最小/最佳)：512 字节 / 512 字节
磁盘标签类型：dos
磁盘标识符：0xe65c76fd

设备       启动  起点       末尾       扇区  大小 Id 类型
/dev/sda1  *     2048 3907027119 3907025072  1.8T  7 HPFS/NTFS/exFAT
root@li:/# 

```

##### du

```shell
root@li:/# du -sh /home/li/*
152G	/home/li/anaconda3
44G	    /home/li/datasets
4.0K	/home/li/Desktop
2.6G	/home/li/Documents
192K	/home/li/Downloads
121M	/home/li/gems
65G	    /home/li/models
13M	    /home/li/Music
397M	/home/li/nltk_data
308K	/home/li/nvvp_workspace
8.0K	/home/li/Oray
124M	/home/li/Pictures
4.0K	/home/li/Public
12M	    /home/li/snap
6.4G	/home/li/software
4.0K	/home/li/Sunlogin Files
104K	/home/li/Templates
21M	    /home/li/Videos
32G	    /home/li/virtualplace
50M	    /home/li/WebstormProjects
268G	/home/li/work
```

##### docker清理

查看 Docker 占用情况：

`docker system df`

清理无用的容器、镜像、卷：

```shell
docker image prune -a

docker container prune

docker volume prune

docker network prune
```

或者一键清理：

`docker system prune -af --volumes`

>
> ⚠️ 注意：这个命令会删除所有未使用的容器、镜像、网络和卷，请谨慎操作！
