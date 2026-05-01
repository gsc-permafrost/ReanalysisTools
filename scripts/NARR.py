from dataclasses import dataclass,field
import os
# import sys
import numpy as np
import pandas as pd
import netCDF4
import urllib.request
import geopandas as gpd
# import argparse
import folium
# import configparser
# from datetime import timedelta
from scipy.interpolate import RBFInterpolator
# import datetime

@dataclass(kw_only=True)
class downloadNARR:
    variableName: str
    date: int
    level: str
    baseURL: str = 'https://downloads.psl.noaa.gov/Datasets/NARR/'
    downloadPath: str = os.path.abspath(os.path.join(os.path.split(__file__)[0],'..','ncFiles'))

    def __post_init__(self):
        if type(self.date) is str:
            self.date = int(self.date)
        if self.level == 'pressure' and len(str(self.date)) != 6:
            exit('Expected date in YYYYMMM format')
        if self.level in ['monolevel'] and len(str(self.date)) != 4:
            exit('Expected date in YYYY format')
        
        self.fileName = f"{self.variableName}.{self.date}.nc"
        self.filePath = os.path.join(self.downloadPath,self.fileName)
        url = f"{self.baseURL}/{self.level}/{self.fileName}"

        if not os.path.isfile(self.filePath):
            print(f'downloading: {url}')
            if not os.path.isdir(self.downloadPath):
                os.makedirs(self.downloadPath)
            urllib.request.urlretrieve(url,self.filePath)
            print(f'saved: {self.filePath}')
            

@dataclass(kw_only=True)
class readNARR(downloadNARR):

    def __post_init__(self):
        super().__post_init__()
        
        self.dataset = netCDF4.Dataset(self.filePath)
        self.lon = np.ma.getdata(self.dataset.variables['lon'][:])
        self.lat = np.ma.getdata(self.dataset.variables['lat'][:])
        self.x = np.ma.getdata(self.dataset.variables['x'][:])
        self.y = np.ma.getdata(self.dataset.variables['y'][:])
        self.time = self.dataset.variables['time']
        self.time = netCDF4.num2date(self.time[:], self.time.units,calendar = 'standard',only_use_cftime_datetimes=False)
        self.time = pd.to_datetime(self.time).tz_localize('UTC')
        self.variable = np.ma.getdata(self.dataset.variables[self.variableName.split('.')[0]][:])

@dataclass(kw_only=True)
class interpolateNARR(readNARR):
    samplePoints: dict = field(default_factory=lambda:{
        'name':[],'lat':[],'lon':[]
    })
    grid_pad: int = 2
    searchDistance: float = 5e4

    def __post_init__(self):
        self.samplePoints = gpd.GeoDataFrame(
            data=self.samplePoints, geometry=gpd.points_from_xy(self.samplePoints['lon'], self.samplePoints['lat']), crs="EPSG:4326"
        )
        super().__post_init__()
        
        # WKT description of the NARR LCC projection
        # Source: https://spatialreference.org/ref/sr-org/8214/
        NARR_LCC = '+proj=lcc +lat_1=50 +lat_0=50 +lon_0=-107 +k_0=1 +x_0=5632642.22547 +y_0=4612545.65137 +a=6371200 +b=6371200 +units=m +no_defs'
        self.samplePoints = self.samplePoints.to_crs(NARR_LCC)
        
        bbox = self.samplePoints.total_bounds
        self.target = np.array([self.samplePoints.geometry.x,self.samplePoints.geometry.y]).T
        searchBuffer = self.samplePoints.buffer(self.searchDistance).geometry[0]
        x,y = np.meshgrid(self.x,self.y)
        x,y = x.flatten(),y.flatten()

        # breakpoint()
        gridPoints = gpd.GeoDataFrame(index = np.arange(0,self.lon.shape[0]*self.lon.shape[1]),geometry=gpd.points_from_xy(x,y))
        gridIndex = gridPoints.index.values.reshape(self.lat.shape)
        selection = gridPoints[gridPoints.within(searchBuffer)]
        key = np.where(np.isin(gridIndex,selection.index.values))

        # breakpoint()
        # xi,yi = np.meshgrid(selection.geometry.x.values,selection.geometry.y.values)
        # xi,yi = xi.flatten(),yi.flatten()
        self.xy = np.array([selection.geometry.x.values,selection.geometry.y.values]).T

        self.variableSelection = self.variable[:,key[0],key[1]]

        self.timeSeries = pd.DataFrame(
            index=self.time,
            data={
                self.variableName:[self.interpolate(self.variable[i,key[0],key[1]])[0] for i in range(self.variableSelection.shape[0])]
                })



        # Make plot of grid cells
        lon_box = self.lon[key[0],key[1]]
        lat_box = self.lat[key[0],key[1]]
        # self.x_bounds = [np.where(self.x<bbox[0])[0][-(1+self.grid_pad)],np.where(self.x>bbox[2])[0][self.grid_pad]]
        # self.y_bounds = [np.where(self.y<bbox[1])[0][-(1+self.grid_pad)],np.where(self.y>bbox[3])[0][self.grid_pad]]
        
        # lon_box = self.lon[self.y_bounds[0]:self.y_bounds[1],self.x_bounds[0]:self.x_bounds[1]]
        # lat_box = self.lat[self.y_bounds[0]:self.y_bounds[1],self.x_bounds[0]:self.x_bounds[1]]

        m = folium.Map(location=[self.samplePoints.lat[0],self.samplePoints.lon[0]])   
        for at,on in zip (lat_box.flatten(),lon_box.flatten()):
            folium.Marker([at, on]).add_to(m)
        folium.CircleMarker([self.samplePoints.lat[0],self.samplePoints.lon[0]],popup=self.samplePoints.name[0]).add_to(m)
        m.save(os.path.join(self.downloadPath,f'{self.samplePoints.name[0]}_{self.variableName}_grid_pts.html'))

        breakpoint()

    def interpolate(self,value,kernel='linear'):
        # Interpolates value from grid (xy) to desired points (coords) using a Radial Bias Function
        # Default behavior is to use a thin plate spline function r**2 * log(r)
        return(RBFInterpolator(self.xy, value, kernel=kernel)(self.target))

interpolateNARR(level='monlevel',variableName='air.2m',date=2025,samplePoints={'name':['SCL'],'lat':[69],'lon':[-135]},grid_pad=0)