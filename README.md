# 智能猪舍系统

## 1、项目简介

本项目基于YOLOv5实现了对猪舍中猪数量的识别，利用烟雾、温湿度、光照、人体红外等模块实时获取猪舍的现场环境数据，并使用MQTT协议将获取的的数据传输到网页显示。当出现火灾、温度过高等异常情况时，网页系统会弹窗提醒，并通过微信发出警报信息。管理员收到警报后，能够在网页端控制灯光、排气扇、喷水器的开关。

## 2、项目组成

项目中YOLOv5识别猪的算法在detect文件夹中，view.py中实现了对猪舍的实时识别。

运行项目：

安装相关环境

终端启动Django项目：python manage.py runserver

注：本项目只含有软件部分的代码

### 3、 项目展示

![项目主页](https://github.com/2111lidongyang/Pigsty-Recognition-System/blob/main/home.png)

