# PyCINRAD

[![Build Status](https://travis-ci.com/CyanideCN/PyCINRAD.svg?branch=master)](https://travis-ci.com/CyanideCN/PyCINRAD)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/932a383368954e8cb37ada9b3d783169)](https://app.codacy.com/app/CyanideCN/PyCINRAD?utm_source=github.com&utm_medium=referral&utm_content=CyanideCN/PyCINRAD&utm_campaign=Badge_Grade_Dashboard)

A python package which handles CINRAD radar data reading and plotting.

读取CINRAD雷达数据，进行相关计算并可视化的模块。

## 安装及初始化设置

### 安装要求

Python 3.5 及以上

Cartopy

Metpy

Shapefile

### 安装方法

```
python setup.py install
```

## 模块介绍

### cinrad.datastruct

构建本模块所使用的数据类型

基本类型: `cinrad.datastruct.Raw`

反射率数据类型: `cinrad.datastruct.R` (base: `cinrad.datastruct.Raw`)

速度数据类型: `cinrad.datastruct.V` (base: `cinrad.datastruct.Raw`)

剖面数据类型: `cinrad.datastruct.Section`

该基本类型包含该要素数据，经纬度数据和雷达其他信息（雷达站名，扫描时间等）

### cinrad.io

读取CINRAD雷达数据。

例子：

```python
from cinrad.io import CinradReader, StandardData
f = CinradReader(your_radar_file) #老版本数据
f = StandardData(your_radar_file) #新版本标准数据
f.reflectivity(elevation_angle_level, data_range) #获取反射率数据（为cinrad.datastruct.R类型）
f.velocity(elevation_angle_level, data_range) #获取速度数据（为cinrad.datastruct.V类型）
f.rhi(azimuth, drange) #获取RHI数据（为cinrad.datastruct.Section类型）
```

### cinrad.utils

提供雷达衍生产品的计算（接受`numpy.ndarray`）。将这些功能独立出来的目的是使得计算程序更加通用，
而不仅仅是能计算此程序读取出来的CINRAD雷达数据。

函数名：
`composite_reflectivity`, `echo_tops`, `vert_integrated_liquid`

### cinrad.easycalc

提供雷达衍生产品的计算（接受`list(cinrad.datastruct.Raw)`）
使用`cinrad.io`读取的数据可直接带入该模块下的函数来计算。

函数名：
`quick_cr`, `quick_et`, `quick_vil`

传入一个包含每个仰角数据的list即可计算。

列表生成示例：
```python
r_list = [f.reflectivity(i, drange) for i in f.angleindex_r]
```

### cinrad.visualize

雷达数据可视化，包括`ppi`和`rhi`，仅接受`cinrad.datastruct.Raw`类型。

例子：

```python
from cinrad.visualize.ppi import PPI
fig = PPI(R, highlight='成都市') #绘制基本反射率图片
fig('D:\\') #传入文件夹路径保存图片
from cinrad.visualize.rhi import RHI
fig = RHI(Section) #绘制RHI
fig('D:\\')
```

如果读取了其他雷达的数据，转换成`cinrad.datastruct.Raw`即可使用此模块画图，详见`example`下的`read_nexrad_level3_velocity.py`
传入的文件路径可以是文件夹路径也可以是文件路径（仅接受以`.png	`结尾的文件路径），如果没有传入路径，程序将会把图片保存在用户目录
（Windows 下称为「个人文件夹」，如 `C:\Users\tom`）下的`PyCINRAD`文件夹。
#### 自定义绘图

`PPI`支持传入自定义`colormap`和`norm`。（分别是`cmap`和`norm`参数）。同时，`nlabel`参数控制色阶条标注的个数
，如果不传入将会设置成10个。如果传入`label`参数的话，程序将会优先使用来标注色阶条。


## 其他

回波顶高及垂直积分液态含水量算法来源：肖艳姣, 马中元, 李中华. 改进的雷达回波顶高、垂直积分液态水含量及其密度算法[J]. 暴雨灾害, 2009, 28(3):20-24.

如果你对这个模块感兴趣，欢迎加入这个模块的开发者行列！

同时，如在使用该脚本中有任何问题和建议，可以提Issue，也可以发邮件给我 274555447@qq.com
