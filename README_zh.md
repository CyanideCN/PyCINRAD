# PyCINRAD

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/cinrad)](https://pepy.tech/project/cinrad)
[![DOI](https://zenodo.org/badge/139155365.svg)](https://zenodo.org/badge/latestdoi/139155365)

Decode CINRAD (China New Generation Weather Radar) data and visualize. 

读取CINRAD雷达数据，进行相关计算并可视化的模块。

**使用交流群：480305660**

**`example`文件夹里有详细的使用示例！**

## 安装

### 安装方法

支持Python 3.5 及以上

```
pip install cinrad
```

或在此页面下载并执行
```
python setup.py install
```

## 模块介绍

### cinrad.io

读取CINRAD雷达数据。

例子：

```python
from cinrad.io import CinradReader, StandardData
f = CinradReader(your_radar_file) #老版本数据
f = StandardData(your_radar_file) #新版本标准数据
f.get_data(tilt, drange, dtype) #获取数据
f.get_raw(tilt, drange, dtype)
```
对于单层RHI数据，传入`get_data`的`tilt`参数将会被设置成0。
`get_raw`方法只会以ndarray的形式返回雷达的数据，不会返回其他的地理信息，因此速度会更快，内存占用更少，在大批量分析数据的时候比较推荐使用此方法。`get_data`返回的数据类型为`xarray.Dataset`，因此可以享受`xarray`模块的便利。

```python
>>> print(data)
<xarray.Dataset>
Dimensions:    (azimuth: 366, distance: 920)
Coordinates:
  * azimuth    (azimuth) float32 0.14084807 0.15812683 ... 0.12601277 0.14381513
  * distance   (distance) float64 0.25 0.5 0.75 1.0 ... 229.2 229.5 229.8 230.0
Data variables:
    ZDR        (azimuth, distance) float64 nan nan nan nan ... nan nan nan nan
    longitude  (azimuth, distance) float64 120.2 120.2 120.2 ... 120.6 120.6
    latitude   (azimuth, distance) float64 35.99 35.99 36.0 ... 38.04 38.04
    height     (azimuth, distance) float64 0.1771 0.1792 0.1814 ... 5.218 5.227
Attributes:
    elevation:        0.48339844
    range:            230
    scan_time:        2020-05-17 11:00:28
    site_code:        Z9532
    site_name:        青岛
    site_longitude:   120.23028
    site_latitude:    35.98861
    tangential_reso:  0.25
    nyquist_vel:      8.37801
    task:             VCP21D
```
例如，可以很轻松的把数据保存成netcdf格式。
```python
>>> data.to_netcdf('1.nc')
```
`xarray`的插值也很简单，例如获取方位角300度，距离180km的数据。
```python
>>> data.interp(azimuth=np.deg2rad(300), distance=180)
<xarray.Dataset>
Dimensions:    ()
Coordinates:
    azimuth    float64 5.236
    distance   int32 180
Data variables:
    ZDR        float64 0.3553
    longitude  float64 118.5
    latitude   float64 36.8
    height     float64 3.6
Attributes:
    elevation:        0.48339844
    range:            230
    scan_time:        2020-05-17 11:00:28
    site_code:        Z9532
    site_name:        青岛
    site_longitude:   120.23028
    site_latitude:    35.98861
    tangential_reso:  0.25
    nyquist_vel:      8.37801
    task:             VCP21D
```

`cinrad.io.StandardData.merge`可以合并单仰角的数据，返回一个完整的体扫文件。


#### 转换为`pyart.core.Radar`类型

`cinrad.io.export.standard_data_to_pyart`可以将`cinrad.io.StandardData`转换为`pyart.core.Radar`。

`example`文件夹里有简单示例。

#### 读取PUP数据和SWAN数据

`cinrad.io.PUP`提供读取PUP数据的功能，读取出来的数据为`cinrad.datastruct.Radial`格式并且可以用来绘制PPI。目前只支持径向类型的数据。
`cinrad.io.SWAN`提供相似的接口来解码SWAN数据。

```python
from cinrad.io import PUP
f = PUP(your_radar_file)
data = f.get_data()
```

#### 读取相控阵雷达数据

`cinrad.io.PhasedArrayData`提供读取相控阵雷达基数据的功能，用法和其他接口非常类似。

```python
from cinrad.io import PhasedArrayData
f = PhasedArrayData(your_radar_file)
data = f.get_data(0, 40, 'REF')
```

### cinrad.utils

提供雷达衍生产品的计算（接受`numpy.ndarray`）。将这些功能独立出来的目的是使得计算程序更加通用，
而不仅仅是能计算此程序读取出来的CINRAD雷达数据。

函数名：
`composite_reflectivity`, `echo_tops`, `vert_integrated_liquid`

计算ET和VIL时，考虑到速度问题，模块提供由cython转换而来的python扩展，可以大大提升速度。如果要使用此扩展，请安装cython以及C编译器，并重新安装此模块。

注：对于当反射率很强时，得到的VIL值可能会很大，这是因为该计算函数没有对强回波进行滤除，程序本身是没有问题的。

### cinrad.calc

提供雷达衍生产品的计算
使用`cinrad.io`读取的数据可直接带入该模块下的函数来计算。

传入一个包含每个仰角数据的list即可计算。

注：当模块使用编译的C扩展的时候提供VIL密度的计算。

列表生成示例：
```python
r_list = [f.get_data(i, drange, 'REF') for i in f.angleindex_r]
# 或者
r_list = list(f.iter_tilt(230, 'REF'))
```
#### VCS

`cinrad.calc.VCS`用于计算任意两点剖面，目前支持所有要素。

示例代码：
```python
import cinrad
from cinrad.visualize import Section
f = cinrad.io.CinradReader(your_radar_file)
rl = [f.get_data(i, drange, 'REF') for i in f.angleindex_r]
vcs = cinrad.calc.VCS(rl)
sec = vcs.get_section(start_cart=(111, 25.5), end_cart=(112, 26.7)) # 传入经纬度坐标
sec = vcs.get_section(start_polar=(115, 350), end_polar=(130, 30)) # 传入极坐标
fig = Section(sec)
fig('D:\\')
```

#### 雷达拼图

`cinrad.calc.GridMapper`可以将不同雷达的扫描数据合并成雷达格点拼图。

#### 水凝物分类

`cinrad.calc.hydro_class`从反射率，差分反射率，协相关系数和差分传播相移率计算出10种水凝物类型。

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

雷达数据可视化，包括`PPI`和`Section`。如果传入的是自定义的数据，需要符合本模块构建`xarray.Dataset`的方式，比如坐标和维度的名字，变量的命名，等等。

示例：

```python
from cinrad.visualize import PPI
fig = PPI(R) #绘制基本反射率图片
fig('D:\\') #传入文件夹路径保存图片
from cinrad.visualize import Section
fig = Section(Slice_) #绘制剖面
fig('D:\\')
```

如果读取了其他雷达的数据，转换成`cinrad.datastruct.Radial`即可使用此模块画图，详见`example`下的`read_nexrad_level3_velocity.py`
传入的文件路径可以是文件夹路径也可以是文件路径（仅接受以`.png`结尾的文件路径），如果没有传入路径，程序将会把图片保存在用户目录
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
|`section`|在`ppi`图中绘制的`Slice_`的数据|
|`style`|背景颜色，可设置为黑色`black`或者白色`white`|
|`add_city_names`|标注城市名|

同时`PPI`类中定义有其他绘图函数：
##### PPI.plot_range_rings(self, _range, color='white', linewidth=0.5, **kwargs)

在PPI图上绘制圆圈。

##### PPI.plot_cross_section(self, data, ymax=None)

在PPI图下方加入VCS剖面图，和`vcs`参数相似，用此函数还可以自定义y轴的范围。

##### PPI.storm_track_info(self, filepath)

在PPI图上叠加PUP的STI产品。

## 相关链接

[利用PyCINRAD处理、显示天气雷达基数据](http://climate2weather.cc/2019/05/12/radar/)

## 引用

如果你在你的论文中使用了本模块，请使用下方的DOI添加引用。

[![DOI](https://zenodo.org/badge/139155365.svg)](https://zenodo.org/badge/latestdoi/139155365)

### 使用本模块绘制图片的论文

1. 上海南汇WSR-88D双偏振天气雷达的生物回波识别与分析 doi: 10.16765/j.cnki.1673-7148.2019.03.015

## 其他

回波顶高及垂直积分液态含水量（密度）算法来源：肖艳姣, 马中元, 李中华. 改进的雷达回波顶高、垂直积分液态水含量及其密度算法[J]. 暴雨灾害, 2009, 28(3):20-24.

水凝物分类算法来源：Dolan, B., S. A. Rutledge, S. Lim, V. Chandrasekar, and M. Thurai, 2013: A Robust C-Band Hydrometeor Identification Algorithm and Application to a Long-Term Polarimetric Radar Dataset. J. Appl. Meteor. Climatol., 52, 2162–2186, https://doi.org/10.1175/JAMC-D-12-0275.1.

如果你对这个模块感兴趣，欢迎加入这个模块的开发者行列！

同时，如在使用该模块时有任何问题和建议，可以提Issue，也可以发邮件给我 274555447@qq.com
