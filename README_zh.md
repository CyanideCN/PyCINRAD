# PyCINRAD

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/932a383368954e8cb37ada9b3d783169)](https://app.codacy.com/app/CyanideCN/PyCINRAD?utm_source=github.com&utm_medium=referral&utm_content=CyanideCN/PyCINRAD&utm_campaign=Badge_Grade_Dashboard)

A python package which handles CINRAD radar data reading and plotting.

读取CINRAD雷达数据，进行相关计算并可视化的模块。

## 安装

### 安装要求

Python 3.5 及以上

Cartopy

Metpy

Shapefile

Pyresample

### 安装方法

在此页面下载并执行
```
python setup.py install
```

## 模块介绍

### cinrad.datastruct

构建本模块所使用的数据类型

径向数据类型：`cinrad.datastruct.Radial`

剖面数据类型: `cinrad.datastruct.Slice_`

格点数据类型：`cinrad.datastruct.Grid`

### cinrad.io

读取CINRAD雷达数据。

例子：

```python
from cinrad.io import CinradReader, StandardData
f = CinradReader(your_radar_file) #老版本数据
f = StandardData(your_radar_file) #新版本标准数据
f.get_data(tilt, drange, dtype) #获取数据
```
对于单层RHI数据，传入`get_data`的`tilt`参数将会被设置成0。

#### 将数据保存为NetCDF格式
```python
f.to_nc(path_to_nc_file)
```

**关于最新的标准数据格式请参考`example`里的`Read standard data.ipynb`**

#### 读取PUP数据和SWAN数据

`cinrad.io.PUP`提供读取PUP数据的功能，读取出来的数据为`cinrad.datastruct.Radial`格式并且可以用来绘制PPI。目前只支持径向类型的数据。
`cinrad.io.SWAN`提供相似的接口来解码SWAN数据。

```python
from cinrad.io import PUP
f = PUP(your_radar_file)
data = f.get_data()
```

### cinrad.utils

提供雷达衍生产品的计算（接受`numpy.ndarray`）。将这些功能独立出来的目的是使得计算程序更加通用，
而不仅仅是能计算此程序读取出来的CINRAD雷达数据。

函数名：
`composite_reflectivity`, `echo_tops`, `vert_integrated_liquid`

计算ET和VIL时，考虑到速度问题，模块提供由cython转换而来的python扩展，可以大大提升速度。如果要使用此扩展，请安装cython以及C编译器，并重新安装此模块。

### cinrad.easycalc

提供雷达衍生产品的计算
使用`cinrad.io`读取的数据可直接带入该模块下的函数来计算。

传入一个包含每个仰角数据的list即可计算。

列表生成示例：
```python
r_list = [f.get_data(i, drange, 'REF') for i in f.angleindex_r]
```
#### VCS

`cinrad.easycalc.VCS`用于计算任意两点剖面，目前支持所有要素。

示例代码：
```python
import cinrad
from cinrad.visualize import Section
f = cinrad.io.CinradReader(your_radar_file)
rl = [f.get_data(i, drange, 'REF') for i in f.angleindex_r]
vcs = cinrad.easycalc.VCS(rl)
sec = vcs.get_section(start_cart=(111, 25.5), end_cart=(112, 26.7)) # 传入经纬度坐标
sec = vcs.get_section(start_polar=(115, 350), end_polar=(130, 30)) # 传入极坐标
fig = Section(sec)
fig('D:\\')
```

### cinrad.correct

提供雷达原数据的校正。

#### cinrad.correct.dealias

利用`pyart`的算法进行速度退模糊。（需要C编译器）

```python
import cinrad
#(文件处理部分省略)
v = f.get_data(1, 230, 'VEL')
v_corrected = cinrad.correct.dealias(v)
```

### cinrad.visualize

雷达数据可视化，包括`PPI`和`Section`以及`RHI`，仅接受`cinrad.datastruct`包含的类型。

例子：

```python
from cinrad.visualize import PPI
fig = PPI(R) #绘制基本反射率图片
fig('D:\\') #传入文件夹路径保存图片
from cinrad.visualize import Section
fig = Section(Slice_) #绘制VCS
fig('D:\\')
from cinrad.visualize import RHI
fig = RHI(rhi) #绘制RHI扫描模式的数据
fig('D:\\')
```

如果读取了其他雷达的数据，转换成`cinrad.datastruct.Radial`即可使用此模块画图，详见`example`下的`read_nexrad_level3_velocity.py`
传入的文件路径可以是文件夹路径也可以是文件路径（仅接受以`.png	`结尾的文件路径），如果没有传入路径，程序将会把图片保存在用户目录
（Windows 下称为「个人文件夹」，如 `C:\Users\tom`）下的`PyCINRAD`文件夹。

#### 自定义绘图

`PPI`支持传入其他参数，总结如下。

|参数|功能|
|:-:|:-:|
|`cmap`|色阶|
|`norm`|色阶范围|
|`nlabel`|色阶条标注个数|
|`label`|色阶条标注|
|`highlight`|地区边界高亮|
|`dpi`|分辨率|
|`extent`|绘图的经纬度范围 e.g. `extent=[90, 91, 29, 30]`|
|`add_slice`|在`ppi`图中加上`vcs`的数据|
|`style`|背景颜色，可设置为黑色`black`或者白色`white`|
|`add_city_names`|标注城市名|

同时`PPI`类中定义有其他绘图函数：
##### PPI.plot_range_rings(self, _range, color='white', linewidth=0.5, **kwargs)

在PPI图上绘制圆圈。

##### PPI.plot_cross_section(self, data, ymax=None)

在PPI图下方加入VCS剖面图，和`vcs`参数相似，用此函数还可以自定义y轴的范围。

##### PPI.storm_track_info(self, filepath)

在PPI图上叠加PUP的STI产品。

## 其他

回波顶高及垂直积分液态含水量算法来源：肖艳姣, 马中元, 李中华. 改进的雷达回波顶高、垂直积分液态水含量及其密度算法[J]. 暴雨灾害, 2009, 28(3):20-24.

如果你对这个模块感兴趣，欢迎加入这个模块的开发者行列！

同时，如在使用该模块时有任何问题和建议，可以提Issue，也可以发邮件给我 274555447@qq.com
