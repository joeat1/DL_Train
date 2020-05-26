# 问题描述

用python中的opencv库画3种飞机，并经过旋转扩充，标注关键点，形成飞机飞机分类与关键点预测数据集”AIRPLANES“。
> opencv 安装：`conda install opencv`

# 要求

+ 飞机颜色要求正确：椭圆用蓝色，长方形用红色，三角形用黄色，并且椭圆机身在最顶层，盖住机翼和尾翼
+ 文件夹名称、文件名（xxx机型_xxx角度）要按要求命名

# 详细描述

数据集”AIRPLANES“包括三个飞行器子类”AIRBUS，FIGHTER，UAV“，每个子类下面有10个样本，每个样本包括一张jpg图片及对应的关键点标注txt文本。

其中，每一个子类目录下有每个类型飞机的多个旋转角度的样本图片（例如“AIRBUS_54.jpg”）,及其对应的关键点标注文件（对应“AIRBUS_54.txt”）。


AIRBUS类：前机翼是个三角形，后机翼是个长方形，机身是椭圆
FIGHTER类：前机翼是个三角形，后机翼是个三角形，机身是椭圆
UAV类：前机翼是个长方形，后机翼是个长方形，机身是椭圆

其中，所有飞机的机身都是椭圆，椭圆中点都是图像中点。椭圆的半径为LENTH_HALF、WIDTH_HALF。

因为椭圆中点为图像中点，故三种飞机的的半机身长度都为：
```
LENTH_HALF=SIZE_IMG//2-LP_xxx[‘Nose’][1]
而对于宽度WIDTH_HALF：
AIRBUS：
WIDTH_BACKWING=SIZE_IMG//25
FIGHTER：
WIDTH_HALF=SIZE_IMG//50
UAV：
WIDTH_HALF=SIZE_IMG//50
AIRBUS的长方形尾翼宽度为：
WIDTH_BACKWING=SIZE_IMG//25
FIGHTER的三角形尾翼顶点Y坐标为：
TOP_BACKWING=SIZE_IMG*7//10
UAV的长方形前机翼和长方形尾翼宽度都为：
WIDTH_WING=SIZE_IMG//25
```

经过旋转变换，旋转后的图像存在img_file文件里，旋转后的坐标存在anno_file
