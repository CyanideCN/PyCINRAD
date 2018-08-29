# PyCINRAD
CINRAD data reader.
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/07b8b5e7259f4cb0948f7cedc862b0e7)](https://www.codacy.com/app/CyanideCN/PyCINRAD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=CyanideCN/PyCINRAD&amp;utm_campaign=Badge_Grade)

A python class which can be used to read CINRAD SA/SB/CB files and plot graphs including PPI and RHI.

读取CINRAD雷达基数据的Python脚本，目前支持SA/SB/CB三种雷达数据格式（部分支持CC雷达和双偏振雷达），具体函数用法请看脚本注释。
该脚本目前还在继续开发中，欢迎提Issue/发PR^_^


### 用法
如要自定义文件保存路径，打开config.ini，编辑键"filepath"对应的值即可

```
from CINRAD_radar import Radar
file = 'Z_RADR_I_Z9576_20180629043900_O_DOR_SA_CAP.bin'
radar = Radar(file)
```
由于SA/SB/CB雷达文件里并未记录站号，程序会尝试从文件名里提取站号然后寻找到对应的地理信息，形如 Z_RADR_I_Z9576_20180629043900_O_DOR_SA_CAP.bin 和 RADA_CHN_DOR_L2_O-Z9558-SA-CAP-20180725084700.bin 这两种形式的文件都是可以自动识别的。注：bz2格式的压缩文件也可直接打开。

如果程序没有读出站号，就会抛出警告:
```
Auto fill radar station info failed, please set code manually
```

这时候则需要手动设置站号。
```
radar.set_code('Z9576')
```

#### 绘制PPI
```
radar.draw_ppi(level, drange, 'r', smooth=True)
```
datatype参数目前支持 'r','v'和'et' 三种，对应反射率，速度和回波顶高。

当datatype为'r'时，smooth参数可以对反射率进行平滑处理。


#### 绘制RHI
```
radar.draw_rhi(azimuth, drange)
```
传入方位角和距离即可绘制。

#### 绘制CINRAD CC雷达反射率数据

由于本程序没有完全支持CC雷达，绘制CC雷达反射率的PPI的步骤稍有不同，需要先手动设置该仰角的度数再绘制。

```
radar = Radar('2018072615.12V', radartype='CC')
radar.set_elevation_angle(0.5)
radar.draw_ppi(0, 230, datatype='r')
```

#### 绘制CINRAD SD(S波段双偏振)雷达数据

目前本程序支持绘制基本反射率和差分反射率，后续产品还在开发中。

```
from CINRAD_radar import DPRadar
radar = DPRadar('Z_RADR_I_Z9210_20180401000152_O_DOR_SA_CAP.bin')
radar.draw_ppi(0, 100, 'REF')
```

REF为基本反射率，ZDR为差分反射率。

PS:如在使用该脚本中有任何问题和建议，可以发邮件给我 274555447@qq.com
