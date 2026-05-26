# PyCINRAD

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/cinrad)](https://pepy.tech/project/cinrad)
[![DOI](https://zenodo.org/badge/139155365.svg)](https://zenodo.org/badge/latestdoi/139155365)

Decode CINRAD (China New Generation Weather Radar) data and visualize.

To check out live examples and docs, visit [pycinrad.cn](https://pycinrad.cn/).

[中文说明](https://github.com/CyanideCN/PyCINRAD/blob/master/README_zh.md)

**`example` folder contains detailed examples!**

## Installation

PyCINRAD supports Python version 3.9 and higher.

```
pip install cinrad
```

You can also download from github page and build from source

```
git clone https://github.com/CyanideCN/PyCINRAD.git
cd PyCINRAD
pip install .
```

## Modules

### cinrad.io

Decode CINRAD radar data.

```python
from cinrad.io import CinradReader, StandardData
f = CinradReader(your_radar_file) #Old version data
f = StandardData(your_radar_file) #New standard data (or phased array data)
f.get_data(tilt, drange, dtype) #Get data
f.get_raw(tilt, drange, dtype)
```

The `get_raw` method returns radar records without other geographic information.

The `get_data` method returns `xarray.Dataset` with radar records, geographic coordinates, and all extra attributes. So, all benefits of `xarray` can be enjoyed. Check xarray [documentation](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.html) for more detailed explanation.

For example, it's very convenient to save data as netcdf format.
```python
>>> data.to_netcdf('1.nc')
```

`xarray` also makes interpolation very convenient.
```python
>>> data.interp(azimuth=np.deg2rad(300), distance=180)
```

For single-tilt data (i.e. files that contain only one elevation angle), `cinrad.io.StandardData.merge` can merge these files to a file contains full volumetric scan.

#### Export data to `Py-ART` defined class

Convert data structure defined in this module into `pyart.core.Radar` is very simple. `cinrad.io.export` has a function `standard_data_to_pyart`, which can take `cinrad.io.StandardData` as input and return `pyart.core.Radar` as output.

`example` folder contains a simple demo about this.

#### Decode PUP data and SWAN data

`cinrad.io.StandardPUP` provides functions to decode Standard PUP(rose) data. The extracted data can be further used to create PPI.

`cinrad.io.SWAN` provides similar interface to decode SWAN data.

```python
from cinrad.io import StandardPUP
f = StandardPUP(your_radar_file)
data = f.get_data()
```

#### Decode phased array radar data

`cinrad.io.PhasedArrayData` provides similar interface to decode level 2 data from phased array radar with old format.

```python
from cinrad.io import PhasedArrayData
f = PhasedArrayData(your_radar_file)
data = f.get_data(0, 40, 'REF')
```

### cinrad.utils

This submodule provides some useful algorithms in radar meteorology. All functions only accept `numpy.ndarray` as input data. This submodule extends the usage of this program, as these functions can accept customized data rather than only the data decoded by `cinrad.io`.

### cinrad.calc

For direct computation of decoded data, `cinrad.calc` provides functions that simplify the process of calculation. For functions contained in this submodule, only a list of reflectivity data is required as the argument.

Code to generate the required list:

```python
r_list = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
# or
r_list = list(f.iter_tilt(230, 'REF'))
```

#### VCS

`cinrad.calc.VCS` provides calculation of vertical cross-section for **all variables**.

```python
import cinrad
from cinrad.visualize import Section
f = cinrad.io.CinradReader(your_radar_file)
rl = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
vcs = cinrad.calc.VCS(rl)
sec = vcs.get_section(start_cart=(111, 25.5), end_cart=(112, 26.7)) # pass geographic coordinates (longitude, latitude)
sec = vcs.get_section(start_polar=(115, 350), end_polar=(130, 30)) # pass polar coordinates (distance, azimuth)
fig = Section(sec)
fig('D:\\')
```

#### Radar mosaic

`cinrad.calc.GridMapper` can merge different radar scans into a cartesian grid, also supports CR.

#### Hydrometeor classification

`cinrad.calc.hydro_class` uses algorithm suggested by Dolan to classify hydrometeors into 10 categories. (Requires REF, ZDR, RHO, and KDP)

### cinrad.correct

This submodule provides algorithms to correct raw radar fields.

#### cinrad.correct.dealias

This function can unwrap the folded velocity using algorithm originated from `pyart`. (needs C compiler)

```python
import cinrad
#(some codes omitted)
v = f.get_data(1, 230, 'VEL')
v_corrected = cinrad.correct.dealias(v)
```

### cinrad.visualize

Visualize the data stored in acceptable format (`cinrad.datastruct`). It also means that you can using customized data to perform visualization, as long as the data is stored as `xarray.Dataset` and constructed by the same protocol (variables naming conventions, data coordinates and dimensions, etc.) For further information about this method, please see the examples contained in `example` folder.

```python
from cinrad.visualize import PPI
fig = PPI(R) #Plot PPI
fig('D:\\') #Pass the path to save the fig
from cinrad.visualize import Section
fig = Section(Slice_) #Plot VCS
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
|`section`|cross-section data to ppi plot|
|`style`|background color:`black`/`white`/`transparent`|
|`add_city_names`|annotate name of city on the plot|

Beside args, class `PPI` has some other auxiliary plotting functions.

##### PPI.plot_range_rings(self, _range, color='white', linewidth=0.5, **kwargs)

Plot range rings on the PPI plot.

##### PPI.plot_cross_section(self, data, ymax=None)

Plot VCS section under the PPI plot.

This function is very similar to `vcs` argument of class `PPI`, but the range of y-axis can be adjusted only by this function.

##### PPI.storm_track_info(self, filepath)

Plot PUP STI product on the current PPI map, including past positions, current position, and forecast positions.

## Gallery

#### PPI reflectivity

![PPI reflectivity](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9735_20180304125031_0.6_230_REF.png)

#### Phased array radar reflectivity

![Phased array radar reflectivity](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/ZGZ02_20200826123326_0.9_40_REF.png)

#### PPI reflectivity combined with cross-section

![PPI reflectivity combined with cross-section](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9735_20180304120845_0.6_230_REF.png)

#### Cross-section

![Cross-section](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9735_20180304004209_VCS_25.5N111E_26.5N112E.png)

#### Cross-section other than reflectivity

![ZDR cross-section](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/Z9574_20190321025715_0.5_230_ZDR_29.47N121.44E_29.4N122.04E.png)

#### RHI reflectivity

![RHI reflectivity](https://raw.githubusercontent.com/CyanideCN/PyCINRAD/master/pictures/XXX_XXX_RHI_299_100_REF.png)

## Citation

If you use PyCINRAD in your paper, please cite PyCINRAD using the DOI below.

[![DOI](https://zenodo.org/badge/139155365.svg)](https://zenodo.org/badge/latestdoi/139155365)

## Papers that use plots generated by `PyCINRAD`

1. Recognition and Analysis of Biological Echo Using WSR-88D Dual-polarization Weather Radar in Nanhui of Shanghai doi: 10.16765/j.cnki.1673-7148.2019.03.015

## Notes

The hydrometeor classfication algorithm comes from Dolan, B., S. A. Rutledge, S. Lim, V. Chandrasekar, and M. Thurai, 2013: A Robust C-Band Hydrometeor Identification Algorithm and Application to a Long-Term Polarimetric Radar Dataset. J. Appl. Meteor. Climatol., 52, 2162–2186, https://doi.org/10.1175/JAMC-D-12-0275.1.

If you are interested in this program, you can join the developers of this program. Any contribution is appreciated!
