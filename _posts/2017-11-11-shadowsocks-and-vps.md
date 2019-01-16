---
layout:     post
title:      "科学上网的终极姿势"
subtitle:   "在 Vultr VPS 上搭建 Shadowsocks"
date:       2017-11-11
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
---

> 迫于回国后, 国内上网困难的压力, 以及作为一个离开Google生活就无法自理的人类. 就目前而言, 传统的vpn一直面临着用着用着哪天就被封了的风险, 所以搭建属于自己的服务器就显得迫在眉睫了

### Shadowsocks

![shadowsocks](http://upload-images.jianshu.io/upload_images/9485-b0d99d196d019ec3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Shadowsocks(ss) 是由 [Clowwindy](https://github.com/Clowwindy) 开发的一款软件，其作用本来是加密传输资料。当然，也正因为它加密传输资料的特性，使得 GFW 没法将由它传输的资料和其他普通资料区分开来（上图），也就不能干扰我们访问那些「不存在」的网站了。

### 服务器的选择

搭建自己的翻墙服务器, 首先要购买一台带有root权限的VPS服务器, 对于国外的服务器, 这里有两个选择一个是 [搬瓦工](https://bandwagonhost.com/), 另一个是 [Vultr](https://www.vultr.com/?ref=7264410)

因为搬瓦工在国内太火所以有些超售严重, 所以在这里还是推荐vultr

Vultr是一家VPS(Virtual private server) 服务器的供应商, 有美国、亚洲、欧洲等多地的 VPS。它家的服务器以性价比高闻名，按时间计费，最低的资费为每月 $5。

<a href="https://www.vultr.com/?ref=7260792"><img src="https://www.vultr.com/media/banner_1.png" width="728" height="90"></a>

#### 流程

1. 注册账号, 通过下面这个链接[Vultr](https://www.vultr.com/?ref=7260792) 登入官网注册一个新账号, 网站目前没有被墙速度还不错
2. 部署VPS,注册完成后,选择一个服务器

![server-choice](/img/in-post/vps/server-choice.jpeg)

3. 为VPS 选择安装系统, 这里选择CentOS 6x64 , 因为相较其他 Linux 系统更适配「[锐速](https://github.com/91yun/serverspeeder)」，一个提高连接 VPS 速度的软件。

   ![sever-os](/img/in-post/vps/sever-os.jpeg)

4. 然后就是选择套餐了, 如果仅用于科学上网的话$5就绰绰有余了

5. 剩下的都可以使用默认值

   ![addtional-feature](/img/in-post/vps/addtional-feature.jpeg)

   > 另外，上图第 6 步中，SSH Keys 的作用是，可以让你登录 VPS 时不用每次手动输密码。若只将其用作 Shadowsocks 服务器，仅需要在配置时登录一次，可以完全忽略它。如果要同时作他用，可参考 [此文](https://www.vultr.com/docs/how-do-i-generate-ssh-keys/) 生成并添加 SSH Key。


### 部署Shadowsocks         

 使用SSH登入来配置服务器, Mac下直接使用terminal, Pc的话可以安装putty.   

 使用下面命令来链接ssh

   ```shell
ssh root@<host>
   ```

  我们这里使用 [teddysun](https://teddysun.com/342.html) 的一键安装脚本。

```shell
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks.sh
chmod +x shadowsocks.sh
./shadowsocks.sh 2>&1 | tee shadowsocks.log
```



### Tcp Fast Open

为了更好的链接速度我们还需要多做几步

用nano编辑器打开下面这个文件

```shell
nano /etc/rc.local
```

然后输入, ctrl + x 退出保存

```shell
echo 3 > /proc/sys/net/ipv4/tcp_fastopen
```

同样的

```shell
nano /etc/sysctl.conf
```

然后再文末输入下面内容, 保存退出

```shell
net.ipv4.tcp_fastopen = 3
```

再打开一个 Shadowsocks 配置文件。

```shell
nano /etc/shadowsocks.json
```

把其中 “fast_open” 一项的 false 替换成 true。

```
"fast_open":true
```

输入下面命令重启Shadowsocks

```shell
/etc/init.d/shadowsocks restart
```



### 开启ServerSpeeder

锐速 ServerSpeeder 是一个 TCP 加速软件，对 Shadowsocks 客户端和服务器端间的传输速度有显著提升。

而且，不同于 FinalSpeed 或 Kcptun 等需要客户端的工具，「锐速」的一大优势是只需要在服务器端单边部署就行了。换句话说，你不需要再安装另外一个应用。另外，「锐速」虽然已经停止注册和安装了，不过网上还是有不少「破解版」可用。

使用一键安装脚本:

```shell
wget -N --no-check-certificate https://raw.githubusercontent.com/91yun/serverspeeder/master/serverspeeder-all.sh && bash serverspeeder-all.sh
```

如果提示不支持该内核:

CentOS 6 更换内核

```shell
rpm -ivh http://soft.91yun.org/ISO/Linux/CentOS/kernel/kernel-firmware-2.6.32-504.3.3.el6.noarch.rpm
rpm -ivh http://soft.91yun.org/ISO/Linux/CentOS/kernel/kernel-2.6.32-504.3.3.el6.x86_64.rpm --force
```

CentOS 7 更换内核

```shell
rpm -ivh http://soft.91yun.org/ISO/Linux/CentOS/kernel/kernel-3.10.0-229.1.2.el7.x86_64.rpm --force
```





至此就搭建成功了

