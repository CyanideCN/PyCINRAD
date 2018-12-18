# PyCINRAD

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/932a383368954e8cb37ada9b3d783169)](https://app.codacy.com/app/CyanideCN/PyCINRAD?utm_source=github.com&utm_medium=referral&utm_content=CyanideCN/PyCINRAD&utm_campaign=Badge_Grade_Dashboard)

A python package which can read CINRAD radar data, perform calculations and visualize the data.

[中文说明](https://github.com/CyanideCN/PyCINRAD/blob/master/README_zh.md)

## Installation

Python 3.5 +

Cartopy

Metpy

Shapefile

Numba

Pyresample

You can directly install this module via

```
pip install cinrad
```

Alternatively, you can download from github page and then excecute

```
python setup.py install
```

## Modules

### cinrad.datastruct

This submodule contains data structure used in this program.

Radial type data: `cinrad.datastruct.Radial`

Cross-section type data: `cinrad.datastruct._Slice`

Grid type data: `cinrad.datastruct.Grid`

### cinrad.io

Decode CINRAD radar data.

Simple demonstration:

```python
from cinrad.io import CinradReader, StandardData
f = CinradReader(your_radar_file) #Old version data
f = StandardData(your_radar_file) #New standard data
f.get_data(tilt, drange, dtype) #Get data
f.rhi(azimuth, drange) #Get range-height indicator data
```

#### Saving data in NetCDF format
```python
f.to_nc(path_to_nc_file)
```

### cinrad.utils

This submodule provides some useful algorithms in radar meteorology. All functions contained only accept `numpy.ndarray`. This submodule extends the usage of this program, as these functions can accept customized data rather than only the data decoded by `cinrad.io`.

### cinrad.easycalc

For directly computation of decoded data, `cinrad.easycalc` provides functions that simplify the process of calculation. For functions contained in this submodule, only a list of reflectivity data is required as the argument.

Code to generate the required list:

```python
r_list = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
```

#### VCS

`cinrad.easycalc.VCS` provides calculation of vertical cross-section.

Sample code
```python
import cinrad
from cinrad.visualize import Section
f = cinrad.io.CinradReader(your_radar_file)
rl = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
vcs = cinrad.easycalc.VCS(rl)
sec = vcs.get_section(start_cart=(111, 25.5), end_cart=(112, 26.7)) # pass geographic coordinates (latitude, longitude)
sec = vcs.get_section(start_polar=(115, 350), end_polar=(130, 30)) # pass polar coordinates (distance, azimuth)
fig = Section(sec)
fig('D:\\')
```

### cinrad.visualize

Visualize the data stored in acceptable format (`cinrad.datastruct`). It also means that you can using customized data to construct a object belongs to one of the class in `cinrad.datastruct` and then perform visualization. For further information about this method, please see the examples contained in `example` folder.

Simple demonstration:

```python
from cinrad.visualize import PPI
fig = PPI(R) #Plot PPI
fig('D:\\') #Pass the path to save the fig
from cinrad.visualize import Section
fig = Section(_Slice) #Plot RHI
fig('D:\\')
```

The path passed into the class can either be the folder path or the file path. Also, if no path is passed, the figure will be saved at the folder named `PyCINRAD` in the home folder (e.g. `C:\Users\tom`).

#### Customize plot settings

The summary of args that can be passed into `PPI` are listed as follows.

|arg|function|
|:-:|:-:|
|`cmap`|colormaps used for plotting|
|`norm`|norm used for plotting|
|`nlabel`|number of labels on the colorbar|
|`label`|labels on the colorbar|
|`highlight`|highlight area of input name|
|`dpi`|dpi of figure|
|`extent`|area to plot e.g. `extent=[90, 91, 29, 30]`|
|`add_slice`|add cross-section data to ppi plot|

Besides args, class `PPI` has some other auxiliary plotting functions.

##### PPI.plot_range_rings(self, _range, color='white', linewidth=0.5, **kwargs)

Plot range rings on the PPI plot.

##### PPI.plot_cross_section(self, data, ymax=None)

Plot VCS section under the PPI plot.

This function is very similar to `vcs` argument of `PPI`, but the range of y-axis can be adjusted only by this function.

## Gallery

#### PPI reflectivity

![PPI reflectivity](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9735_20180304125031_0.6_230_REF.png)

#### PPI reflectivity combined with cross-section

![PPI reflectivity combined with cross-section](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9735_20180304120845_0.6_230_REF.png)

#### Cross-section

![Cross-section](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9735_20180304004209_VCS_25.5N111E_26.5N112E.png)

## Notes

If you are interested in this program, you can join the developers of this program. Any contribution is appreciated!

If you have some questions or advise about this program, you can create a issue or email me at 274555447@qq.com.
