# -*- coding: utf-8 -*-
# Author: Puyuan Du

import warnings
from typing import Union

from netCDF4 import Dataset
from numpy import ndarray

from cinrad.datastruct import Radial, Grid # Coding convenience
from cinrad._typing import number_type

prodname = {'REF':'Base Reflectivity', 'VEL':'Base Velocity', 'CR':'Composite Reflectivity',
            'ET':'Echo Tops', 'VIL':'Vertically Integrated Liquid', 'ZDR':'Differential Reflectivity',
            'PHI':'Difference Phase', 'RHO':'Correlation Coefficient'}

class NetCDFWriter:
    def __init__(self, filepath:str):
        self.da = Dataset(filepath, 'w', format='NETCDF4')
        self.group = dict()

    def _create_group(self, groupname:str):
        if groupname in self.da.groups.keys():
            return
        gp = self.da.createGroup(groupname)
        self.group[groupname] = gp

    def _create_dimension(self, groupname:str, dimension:str, shape:tuple):
        gp = self.group[groupname]
        if dimension in gp.dimensions:
            return
        gp.createDimension(dimension, shape)

    def _create_variable(self, groupname:str, varname:str, variable:ndarray, dimension:Union[tuple, str],
                         datatype:str='f8'):
        gp = self.group[groupname]
        if varname in gp.variables.keys():
            return
        if isinstance(dimension, str):
            if dimension not in gp.dimensions:
                raise ValueError('Dimension {} not created'.format(dimension))
        elif isinstance(dimension, tuple):
            for i in dimension:
                if i not in gp.dimensions:
                    raise ValueError('Dimension {} not created'.format(dimension))
        gp.createVariable(varname, datatype, dimension)
        gp.variables[varname][:] = variable

    def _create_attribute(self, attrname:str, value:number_type):
        self.da.setncattr(attrname, value)

    def close(self):
        self.da.close()

    def create_radial(self, radial:Radial):
        el = radial.elev.round(2)
        gpname = 'Elevation angle {}'.format(el)
        self._create_group(gpname)
        self._create_dimension(gpname, 'Azimuth', radial.az.shape[0])
        self._create_dimension(gpname, 'Distance', radial.dist.shape[0])
        self._create_variable(gpname, 'Azimuth', radial.az, 'Azimuth')
        self._create_variable(gpname, 'Distance', radial.dist, 'Distance')
        self._create_variable(gpname, 'Longitude', radial.lon, ('Azimuth', 'Distance'))
        self._create_variable(gpname, 'Latitude', radial.lat, ('Azimuth', 'Distance'))
        if radial.dtype == 'VEL':
            warnings.warn('Velocity Data is currently not supported', RuntimeWarning)
            return
            # self._create_variable(gpname, 'Base Velocity', radial.data[0], ('Azimuth', 'Distance'))
            # self._create_variable(gpname, 'Range Fold', radial.data[1], ('Azimuth', 'Distance'))
        else:
            self._create_variable(gpname, prodname[radial.dtype], radial.data, ('Azimuth', 'Distance'))

    def create_grid(self, grid:Grid):
        gpname = 'Derived field'
        self._create_group(gpname)
        self._create_dimension(gpname, 'Longitude', grid.lon.shape[0])
        self._create_dimension(gpname, 'Latitude', grid.lat.shape[1])
        self._create_variable(gpname, 'Longitude', grid.lon[0], 'Longitude')
        self._create_variable(gpname, 'Latitude', grid.lat[:, 0], 'Latitude')
        self._create_variable(gpname, prodname[grid.dtype], grid.data, ('Longitude', 'Latitude'))