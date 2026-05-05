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
    variableNames: list
    dates: list
    level: str
    baseURL: str = 'https://downloads.psl.noaa.gov/Datasets/NARR'
    downloadPath: str = os.path.abspath(os.path.join(os.path.split(__file__)[0],'..','ncFiles'))

    def __post_init__(self):
        if not isinstance(self.dates,list):
            self.dates = [self.dates]
            for i,d in enumerate(self.dates):
                if type(d) is str:
                    self.dates[i] = int(d)
                if self.level == 'pressure' and len(str(d)) != 6:
                    exit('Expected date in YYYYMMM format')
                if self.level in ['monolevel'] and len(str(d)) != 4:
                    exit('Expected date in YYYY format')
        if not isinstance(self.variableNames,list):
            self.variableNames = [self.variableNames]
        
        self.fileNames = {}
        self.filePaths = {}
        self.fileIndex = {d:{vn:f"{vn}.{d}.nc" for vn in self.variableNames} for d in self.dates}
        for y,vars in self.fileIndex.items():
            for vn,fn in vars.items():
                fp,exits = self.filePath(fn)
                if not exits:
                    url = self.urlPath(fn)
                    print(f'downloading: {url}')
                    if not os.path.isdir(self.downloadPath):
                        os.makedirs(self.downloadPath)
                    urllib.request.urlretrieve(url,fp)
                    print(f'saved: {fp}')
        # self.fileIndex = [[vn.split('.')[0],d] for vn in self.variableNames for d in self.dates] 
        # self.fileNames = [f"{vn}.{d}.nc" for vn in self.variableNames for d in self.dates]
        # self.filePaths = [os.path.join(self.downloadPath,fn) for fn in self.fileNames]
        # urls = [f"{self.baseURL}/{self.level}/{fn}" for fn in self.fileNames]
        # for i,fp in enumerate(self.filePaths):
        #     if not os.path.isfile(fp):
        #         print(f'downloading: {urls[i]}')
        #         if not os.path.isdir(self.downloadPath):
        #             os.makedirs(self.downloadPath)
        #         urllib.request.urlretrieve(urls[i],fp)
        #         print(f'saved: {fp}')

    def filePath(self,fn):
        fp = os.path.join(self.downloadPath,fn)
        return(fp,os.path.isfile(fp))
    
    def urlPath(self,fn):
        return(f"{self.baseURL}/{self.level}/{fn}")
            
@dataclass(kw_only=True)
class readNARR(downloadNARR):

    def __post_init__(self):
        super().__post_init__()
        self.variables = {}
        time = []
        variables = [[] for v in self.variableNames]
        j = 0
        
        self.coordinates = {}
        self.data = {}
        self.metadata = {}
        # coordinates = {'lon':{},'lat':{},'x':{},'y':{}}
        i = 0
        for yr,vars in self.fileIndex.items():
            for vn,fn in vars.items():
                fp,exits = self.filePath(fn)
                if not exits:
                    exit()
                dataset = netCDF4.Dataset(fp)
                if vn == self.variableNames[0]:
                    tx = dataset.variables['time']
                    tx = netCDF4.num2date(tx[:], tx.units,calendar = 'standard',only_use_cftime_datetimes=False)
                    if yr == self.dates[0]:
                        index = tx
                    else:
                       index = np.concatenate((index,tx))
                if yr == self.dates[0]:
                    self.coordinates[vn] = {}
                    self.coordinates[vn]['lon'] = np.ma.getdata(dataset.variables['lon'][:])
                    self.coordinates[vn]['lat'] = np.ma.getdata(dataset.variables['lat'][:])
                    self.coordinates[vn]['x'] = np.ma.getdata(dataset.variables['x'][:])
                    self.coordinates[vn]['y'] = np.ma.getdata(dataset.variables['y'][:])
                    self.metadata[vn] = dataset.variables[vn.split('.')[0]]
                    self.data[vn] = []
                    self.data[vn] = np.ma.getdata(dataset.variables[vn.split('.')[0]][:])
                else:
                    self.data[vn] = np.concatenate((
                        self.data[vn],np.ma.getdata(dataset.variables[vn.split('.')[0]][:])
                    ))
        self.index = pd.to_datetime(index).tz_localize('UTC')

@dataclass(kw_only=True)
class interpolateNARR(readNARR):
    samplePoints: dict = field(default_factory=lambda:{
        'name':[],'lat':[],'lon':[]
    })
    grid_pad: int = 2
    searchDistance: float = 5e4
    extrapolate: bool = False

    def __post_init__(self):
        self.samplePoints = gpd.GeoDataFrame(
            data=self.samplePoints, geometry=gpd.points_from_xy(self.samplePoints['lon'], self.samplePoints['lat']), crs="EPSG:4326"
        )
        
        # WKT description of the NARR LCC projection
        # Source: https://spatialreference.org/ref/sr-org/8214/
        NARR_LCC = '+proj=lcc +lat_1=50 +lat_0=50 +lon_0=-107 +k_0=1 +x_0=5632642.22547 +y_0=4612545.65137 +a=6371200 +b=6371200 +units=m +no_defs'
        self.samplePoints = self.samplePoints.to_crs(NARR_LCC)
        
        bbox = self.samplePoints.total_bounds
        self.target = np.array([self.samplePoints.geometry.x,self.samplePoints.geometry.y]).T
        searchBuffer = self.samplePoints.buffer(self.searchDistance).geometry[0]


        super().__post_init__()
        
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

        self.dataSelection = {var:self.data[var][:,key[0],key[1]] for var in self.variableNames}

        self.timeSeries = pd.concat(
            [pd.DataFrame(
                index=self.index,
                columns=[f"{vn}_{n}" for n in self.samplePoints['name']],
                data = np.array(
                    [self.interpolate(self.data[vn][i,key[0],key[1]]) for i in range(self.index.shape[0])]
                )
            )
            for vn in self.variableNames],axis=1
        )
        breakpoint()

        # Make plot of grid cells
        # lon_box = self.lon[key[0],key[1]]
        # lat_box = self.lat[key[0],key[1]]

        # m = folium.Map(location=[self.samplePoints.lat[0],self.samplePoints.lon[0]])   
        # for at,on in zip (lat_box.flatten(),lon_box.flatten()):
        #     folium.Marker([at, on]).add_to(m)
        # folium.CircleMarker([self.samplePoints.lat[0],self.samplePoints.lon[0]],popup=self.samplePoints.name[0]).add_to(m)
        # m.save(os.path.join(self.downloadPath,f'{self.samplePoints.name[0]}_{self.variableNames}_grid_pts.html'))

        # breakpoint()

    def interpolate(self,value,kernel='linear'):
        # Interpolates value from grid (xy) to desired points (coords) using a Radial Bias Function
        # Default behavior is to use a thin plate spline function r**2 * log(r)

        vx = RBFInterpolator(self.xy, value, kernel=kernel)(self.target)
        if not self.extrapolate:
            vx[vx<value.min()]=value.min()
            vx[vx>value.max()]=value.max()
        return(vx)
