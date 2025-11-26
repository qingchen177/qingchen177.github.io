---
layout: post
title: Fedora记录
tags: [Fedora,向日葵,PyCharm]
categories: [操作系统]
---
## Fedora

> 记录一下我简短的Fedora使用记录吧。
>
> 8年的笔记本长时间没用，再次开机很卡，看了一下i3 4核CPU，8G内存， windows是扛不住了，然后当时想换Centos，但是又停止维护，就换了Fedora。
>
> 也是折腾了一段时间，现在换工作了，把这台笔记本又用上了，但是我现在用的是Ubuntu，时不时换Fedora安装总是敲错，md干脆直接都换Ubuntu了。
>
> 细节肯定是记不清了，趁着在重装系统之前，备份文件看看有没有什么要记录的，印象最深的是安装向日葵的时候改了东西，RDP我真不习惯，每次都注销用户下次要重新打开都不知道上次远程干的啥了。

`sudo neofetch`（`sudo yum install neofetch`）

```shell
             .',;::::;,'.                root@qingchen 
         .';:cccccccccccc:;,.            ------------- 
      .;cccccccccccccccccccccc;.         OS: Fedora Linux 40 (Workstation Edition) x86_64 
    .:cccccccccccccccccccccccccc:.       Host: MRC-WX0 M1Q 
  .;ccccccccccccc;.:dddl:.;ccccccc;.     Kernel: 6.12.6-100.fc40.x86_64 
 .:ccccccccccccc;OWMKOOXMWd;ccccccc:.    Uptime: 1 hour, 18 mins 
.:ccccccccccccc;KMMc;cc;xMMc:ccccccc:.   Packages: 2336 (rpm), 13 (flatpak) 
,cccccccccccccc;MMM.;cc;;WW::cccccccc,   Shell: bash 5.2.26 
:cccccccccccccc;MMM.;cccccccccccccccc:   Resolution: 1920x1080 
:ccccccc;oxOOOo;MMM0OOk.;cccccccccccc:   WM: Mutter 
cccccc:0MMKxdd:;MMMkddc.;cccccccccccc;   WM Theme: Adwaita 
ccccc:XM0';cccc;MMM.;cccccccccccccccc'   Theme: Adwaita [GTK2/3] 
ccccc;MMo;ccccc;MMW.;ccccccccccccccc;    Icons: Adwaita [GTK2/3] 
ccccc;0MNc.ccc.xMMd:ccccccccccccccc;     Terminal: gnome-terminal 
cccccc;dNMWXXXWM0::cccccccccccccc:,      CPU: Intel i3-7020U (4) @ 2.300GHz 
cccccccc;.:odl:.;cccccccccccccc:,.       GPU: Intel HD Graphics 620 
:cccccccccccccccccccccccccccc:'.         Memory: 4133MiB / 7723MiB 
.:cccccccccccccccccccccc:;,..
  '::cccccccccccccc::;,.
```

其实大部分和Ubuntu没区别，界面安装折腾一样一样的

主要就是

```shell
# Fedora
yum install xxx
dnf install xxx.rpm
rpm -ivh xxx.rpm
# Ubuntu
apt install xxx
dpkg -i xxx.deb
```

就记得两个问题写一下吧

----

### pycharm部分中文乱码

参考：

1. https://www.bilibili.com/opus/802918040027004934?spm_id_from=333.976.0.0
2. https://bugzilla.redhat.com/show_bug.cgi?id=2188765

解决（我记得是这样就直接解决了）

```shell
sudo dnf install google-noto-sans-cjk-fonts.noarch --allowerasing
sudo dnf install google-noto-sans-cjk-ttc-fonts
```

-----

### 向日葵安装

参考：https://tieba.baidu.com/p/7157359656

我这里就是按照贴吧做的

```shell
cd /usr/local/sunlogin/
# 修改
vim rpminstall.sh
vim script/common.sh
vim script/start.sh
```

应该是中途要装个这个，记不清楚了，反正你们可以先不装遇到报错了去装这个

https://fedora.pkgs.org/40/fedora-x86_64/libxcrypt-compat-4.4.36-5.fc40.x86_64.rpm.html

好吧啥也想不起来直接贴源文件吧

`qingchen@qingchen:/usr/local/sunlogin$ cat rpminstall.sh` 

```shell
#!/bin/bash

#change directory to script path
curpath=$(cd "$(dirname "$0")"; pwd)
cd $curpath > /dev/null

source /usr/local/sunlogin/scripts/common.sh

#kill all runing sunloginclient
killall sunloginclient
killall sunloginclient_linux

#clear log files
if [ -d '/var/log/sunlogin' ]; then
  rm -rf /var/log/sunlogin
fi
if [ -d "/var/log/sunloginent" ] ; then
  rm /var/log/sunloginent/*  > /dev/null 2>&1
fi
mkdir /var/log/sunlogin
chmod 777 /var/log/sunlogin


if [ $os_name == 'centos' ] || [ $os_name == 'fedora' ] || [ "$(echo $os_name |grep redhat)" != "" ] || [ $os_name == 'nfs_server' ]; then
    echo 'check operate system OK'
else
    echoAndExit 'unknown OS it not impl'
fi
    
os_version_int=${os_version%.*}

chmod 755 /usr/local/sunlogin/sunlogin.desktop
cp /usr/local/sunlogin/sunlogin.desktop /usr/share/applications/
chmod 644 /usr/local/sunlogin/res/skin/*.skin
chmod 644 /usr/share/applications/sunlogin.desktop
chmod 644 /usr/local/sunlogin/res/icon/*

#echo "create init"

if  [ "$os_name" == "centos" ] || [ $os_name == 'fedora' ] || [ "$(echo $os_name |grep redhat)" != "" ] ; then

    if [ "$os_version_int" -lt 7 ]; then
        cp /usr/local/sunlogin/scripts/init_runsunloginclient /etc/init.d/runsunloginclient || echoAndExit 'can not copy init file init_runsunloginclient'
        chmod +x /etc/init.d/runsunloginclient
        #create soft link    
        for i in $(seq 0 6)
        do
            ln -s /etc/init.d/runsunloginclient /etc/rc$i.d/S99runsunloginclient > /dev/null 2>&1
        done
        /sbin/chkconfig --add runsunloginclient
        /sbin/chkconfig runsunloginclient on
    else
        cp /usr/local/sunlogin/scripts/runsunloginclient.service /etc/systemd/system/runsunloginclient.service || echoAndExit 'can not copy init file runsunloginclient.service'
        systemctl enable runsunloginclient.service
    fi
elif [ "$os_name" == "nfs_server" ] ; then

    cp /usr/local/sunlogin/scripts/runsunloginclient.service /etc/systemd/system/runsunloginclient.service || echoAndExit 'can not copy init file runsunloginclient.service'
    systemctl enable runsunloginclient.service
else
    echo 'unknown OS is not impl'
fi
/usr/local/sunlogin/scripts/start.sh
```

`qingchen@qingchen:/usr/local/sunlogin$ cat scripts/common.sh` 

```shell
function echoAndExit
{
    echo -n 'Error:'
    echo $1
    echo 'Installation failed'
    exit 1
}

#check root
function check_root
{
    if [ $(whoami) != 'root' ]; then
        if [ "$1" == "" ]; then
            echo 'Sunlogin client needs root to complete installation'
        else
            echo "$1"
        fi
        exit 1
    fi
}

function killallsunloginclient
{
    if [[ $(ps -A | grep sunloginclient) != "" ]]; then
        killall sunloginclient
    fi
}

path_main='/usr/local/sunlogin'
path_bin="$path_main/bin"
path_etc="$path_main/etc"
path_doc="$path_main/doc"
path_log="$path_main/var/log"

#get operation system info
function get_os_name()
{
    if grep -Eqii "CentOS" /etc/issue || grep -Eq "CentOS" /etc/*-release; then
        DISTRO='centos'
        PM='yum'
    elif grep -Eqi "Red Hat Enterprise Linux Server" /etc/issue || grep -Eq "Red Hat Enterprise Linux Server" /etc/*-release; then
        DISTRO='redhat'
        PM='yum'
    elif grep -Eqi "Aliyun" /etc/issue || grep -Eq "Aliyun" /etc/*-release; then
        DISTRO='Aliyun'
        PM='yum'
    elif grep -Eqi "Fedora" /etc/issue || grep -Eq "Fedora" /etc/*-release; then
        DISTRO='Fedora'
        PM='yum'
    elif grep -Eqi "Debian" /etc/issue || grep -Eq "Debian" /etc/*-release; then
        DISTRO='Debian'
        PM='apt'
    elif grep -Eqi "Deepin" /etc/issue || grep -Eq "Deepin" /etc/*-release; then
        DISTRO='Deepin'
        PM='apt'
    elif grep -Eqi "Ubuntu" /etc/issue || grep -Eq "Ubuntu" /etc/*-release; then
        DISTRO='ubuntu'
        PM='apt'
    elif grep -Eqi "Raspbian" /etc/issue || grep -Eq "Raspbian" /etc/*-release; then
        DISTRO='Raspbian'
        PM='apt'
    elif grep -Eqi "Kylin" /etc/issue || grep -Eq "Kylin" /etc/*-release; then
        DISTRO='kylin'
        PM='apt'
    elif grep -Eqi "uos" /etc/issue || grep -Eq "uos" /etc/*-release; then
        DISTRO='Deepin'
        PM='apt'
    elif grep -Eqi "NFS Server" /etc/issue || grep -Eq "NFS Server" /etc/*-release; then
        DISTRO='nfs_server'
        PM='yum'
    elif grep -Eqi "方德桌面" /etc/issue || grep -Eq "方德桌面" /etc/*-release; then
        DISTRO='nfs_desktop'
        PM='apt'
    elif grep -Eqi "Loongnix" /etc/issue || grep -Eq "Loongnix" /etc/*-release; then
        DISTRO='Loongnix'
        PM='apt'
    else
        DISTRO='unknow'
    fi
    echo $DISTRO;
}

os_name=$(get_os_name)
echo $os_name
os_version='0.0'

if [ $os_name == 'ubuntu' ]; then
    os_version=`cat /etc/issue | cut -d' ' -f2`
elif [ $os_name == 'kylin' ]; then
    os_version=`cat /etc/issue | cut -d' ' -f2`
elif [ $os_name == 'Deepin' ]; then
    os_version=`cat /etc/lsb-release |grep DISTRIB_RELEASE | cut -d'=' -f2 |sed 's/"//g'`
elif  [ "$os_name" == "centos" ] || [ "$(echo $os_name |grep redhat)" != "" ] ; then
    os_version=`rpm -q centos-release|cut -d- -f3`
elif  [ "$os_name" == "fedora" ]; then
    os_version=`rpm -q fedora-release|cut -d- -f3`
elif [ $os_name == 'nfs_server' ] || [ $os_name == 'nfs_desktop' ]; then
    os_version=`cat /etc/os-release |grep VERSION_ID | cut -d'=' -f2 |sed 's/"//g'`
elif [ $os_name == 'Loongnix' ]; then
    os_version=`cat /etc/issue | cut -d' ' -f3`
fi

os_name=$(echo $os_name | tr [A-Z] [a-z])
os_bits=$(getconf LONG_BIT)


srv_type="initd"
which systemctl >/dev/null 2>&1
if [ $? -eq 0 ]; then
    srv_type="systemd"
else
    srv_type="initd"
fi


#echo $os_name
#echo $os_version
```

`qingchen@qingchen:/usr/local/sunlogin$ cat scripts/start.sh` 

```shell
#!/bin/bash
isinstalled=true
ipaddress=''
oray_vpn_address=''
isinstalledcentos()
{
if [ -a "/etc/init.d/runsunloginclient" ]; then
    echo "Installed" > /dev/null
else
    echo "Please run install.sh first"
    isinstalled=false
    exit
fi
}

isinstalledubuntu()
{
if [ -a "/etc/init/runsunloginclient.conf" ]; then
    echo "Installed" > /dev/null
else
    echo "Please run install.sh first"
    isinstalled=false
    exit
fi
}

isinstalledubuntu_hv()
{
if [ -a "/etc/systemd/system/runsunloginclient.service" ]; then
    echo "Installed" > /dev/null
else
    echo "Please run install.sh first"
    isinstalled=false
    exit
fi
}

isinstalledcentos_hv()
{
if [ -a "/etc/systemd/system/runsunloginclient.service" ]; then
    echo "Installed" > /dev/null
else
    echo "Please run install.sh first"
    isinstalled=false
    exit
fi
}

#change directory to script path
curpath=$(cd "$(dirname "$0")"; pwd)
cd $curpath > /dev/null

source /usr/local/sunlogin/scripts/common.sh
os_version_int=${os_version%.*}
for i in $(seq 1 10)
do
    os_version_int=${os_version_int%.*}
done

#check root
check_root "Installed Sunlogin client needs root to start"
ifconfig_bin='ifconfig'

if [ $os_name == 'ubuntu' ]; then
    if [ $isinstalled == true ]; then
        if [ -n "$os_version_int" ] && [ $os_version_int -lt 15 ]; then
            isinstalledubuntu
            initctl start runsunloginclient --system
        else
            isinstalledubuntu_hv
            systemctl start runsunloginclient.service
        fi
    fi
elif [ $os_name == 'kylin' ]; then
    if [ $isinstalled == true ]; then
        isinstalledubuntu_hv
        systemctl start runsunloginclient.service
    fi
elif [ $os_name == 'deepin' ]; then
    if [ $isinstalled == true ]; then
        if [ -n "$os_version_int" ] && [ $os_version_int -gt 2000 ]; then
            let os_version_int=os_version_int-2000
        fi
        if [ -n "$os_version_int" ] && [ $os_version_int -lt 15 ]; then
            isinstalledubuntu
            initctl start runsunloginclient --system
        else
            isinstalledubuntu_hv
            systemctl start runsunloginclient.service
        fi
    fi
elif  [ "$os_name" == "centos" ] || [ "$os_name" == "fedora" ] || [ "$(echo $os_name |grep redhat)" != "" ] ; then
    if [ -n "$os_version_int" ] && [ $os_version_int -lt 7 ]; then
        isinstalledcentos
        if [ $isinstalled == true ]; then
            ifconfig_bin='/sbin/ifconfig'
            #Proactively stop the service during overwriting installation
            /sbin/service runsunloginclient stop
            #need to wait for the service to completely exit and wait for 1 second
            sleep  1
            #/sbin/service iptables stop
            /sbin/service runsunloginclient start
        fi
    else
        isinstalledcentos_hv
        #Proactively stop the service during overwriting installation
        systemctl stop runsunloginclient.service
        #need to wait for the service to completely exit and wait for 1 second
        sleep  1
        #systemctl stop firewalld.service
        systemctl start runsunloginclient.service
    fi
elif [ $os_name == 'nfs_desktop' ] || [ $os_name == 'nfs_server' ]; then
    if [ $isinstalled == true ]; then
        isinstalledubuntu_hv
        systemctl start runsunloginclient.service
    fi
elif [ $os_name == 'loongnix' ]; then
    if [ $isinstalled == true ]; then
        isinstalledubuntu_hv
        systemctl start runsunloginclient.service
    fi
elif [ $srv_type == 'systemd' ]; then
    if [ $isinstalled == true ]; then
        isinstalledubuntu_hv
        systemctl start runsunloginclient.service
    fi
elif [ $srv_type == 'initd' ]; then
    if [ $isinstalled == true ]; then
        isinstalledubuntu
        initctl start runsunloginclient --system
    fi
fi
#/usr/local/sunlogin/scripts/host > /dev/null 2>&1
#cd - > /dev/null
```

应该是ok的

----

## 相关链接

- Fedora基本安装及轻办公场景下的正常使用笔记（https://zhuanlan.zhihu.com/p/698515004）

----

## END

Fedora有缘再会了！
