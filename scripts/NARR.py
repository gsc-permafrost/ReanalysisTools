from dataclasses import dataclass,field
import os
import yaml
import shutil
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
import datetime
import time

@dataclass(kw_only=True)
class narrData:
    years: list
    variableNames: list
    level: str = 'monolevel'
    baseURL: str = 'https://downloads.psl.noaa.gov/Datasets/NARR'
    downloadPath: str = os.path.abspath(os.path.join(os.path.split(__file__)[0],'..','ncFiles'))
    metadata: dict = field(init=False,default_factory=dict)
    # WKT description of the NARR LCC projection, source: https://spatialreference.org/ref/sr-org/8214/
    NARR_LCC = '+proj=lcc +lat_1=50 +lat_0=50 +lon_0=-107 +k_0=1 +x_0=5632642.22547 +y_0=4612545.65137 +a=6371200 +b=6371200 +units=m +no_defs'

    def __post_init__(self):
        mdF = os.path.join(self.downloadPath,'metadata.yml')
        if os.path.isfile(mdF):
            with open(mdF) as f:
                mdIn = yaml.safe_load(f)
        else:
            mdIn = {}
        self.findFiles()
        cleanMD = {key:{k:v.item() if isinstance(v,np.generic) else v if not isinstance(v,np.ndarray) else v.tolist() for k,v in value.items()} for key,value in self.metadata.items()}
        self.metadata = mdIn | cleanMD
        with open(mdF,'w+') as f:
            yaml.safe_dump(self.metadata,f)

    def findFiles(self):
        
        if not isinstance(self.variableNames,list):
            self.variableNames = [self.variableNames]
        self.fileIndex = {vn:{yr:self.getFile(yr,vn) for yr in self.years} for vn in self.variableNames}
    
    def getFile(self,year,variableName,subFolder='fullYear'):
        filePath,exists = self.formatFilePath(f"{variableName}.{year}.nc",subFolder)
        if not exists:
            if subFolder == 'fullYear':
                filePath = self.getFile(year,variableName,subFolder='partialYear')
            else: 
                self.download(f"{variableName}.{year}.nc",filePath)
                filePath,flag=self.timeCheck(variableName,filePath)
        elif exists and subFolder == 'partialYear':
            filePath,flag=self.timeCheck(variableName,filePath)
            if flag:
                filePath = self.getFile(year,variableName,subFolder='partialYear')
        if variableName not in self.metadata:
            _,md=self.read(variableName,filePath,mode='partial')
            self.metadata[variableName] = md
        return filePath
    
    def timeCheck(self,variableName,filePath):
        tx,md=self.read(variableName,filePath,mode='partial')
        if variableName not in self.metadata:
            self.metadata[variableName] = md
        if tx.month.max()==12 and 'partialYear' in filePath:
            fp = filePath.replace('partialYear','fullYear')
            if not os.path.isdir(os.path.split(fp)[0]):
                os.makedirs(os.path.split(fp)[0])
            shutil.move(filePath,fp)
            filePath = fp
            flag = False
        else:
            # redownload partials over one month old if hasn't been redownloaded in last 24hr to check for updates
            if tx.month.max() != datetime.datetime.now().month-1 and (time.time()-os.path.getctime(filePath))/(3600*24) >1:
                os.remove(filePath)
                print(f"removing and re-downloading: {filePath}")
                flag = True
            else:
                flag = False
        return(filePath,flag)
                
    def getTime(self,dataset):
        tx = dataset.variables['time']
        tx = netCDF4.num2date(tx[:], tx.units,calendar = 'standard',only_use_cftime_datetimes=False)
        return pd.to_datetime(tx).tz_localize('UTC')

    def formatFilePath(self,fn,subFolder=''):
        fp = os.path.join(self.downloadPath,subFolder,fn)
        return(fp,os.path.isfile(fp))
    
    def urlPath(self,fn):
        return(f"{self.baseURL}/{self.level}/{fn}")
    
    def download(self,fileName,filePath):
        url = self.urlPath(fileName)
        print(f'downloading: {url}')
        if not os.path.isdir(os.path.split(filePath)[0]):
            os.makedirs(os.path.split(filePath)[0])
        
        urllib.request.urlretrieve(url,filePath)
        print(f'saved: {filePath}')
    
    def read(self,variableName,filePath,mode='full'):
        if mode != 'full':
            with netCDF4.Dataset(filePath) as dataset:
                tx = self.getTime(dataset)
                md = dataset.variables[variableName.split('.')[0]].__dict__
                return (tx,md)
        with netCDF4.Dataset(filePath) as dataset:
            tx = self.getTime(dataset)
            lon = np.ma.getdata(dataset.variables['lon'][:])
            lat = np.ma.getdata(dataset.variables['lat'][:])
            x = np.ma.getdata(dataset.variables['x'][:])
            y = np.ma.getdata(dataset.variables['y'][:])
            data = np.ma.getdata(dataset.variables[variableName.split('.')[0]][:])
            
            x,y = np.meshgrid(x,y)
            x,y = x.flatten(),y.flatten()
            xy = np.array([x,y]).T
        return(tx,xy,data)


@dataclass(kw_only=True)
class pointEstimates(narrData):
    samplePoints: gpd.GeoDataFrame = field(default_factory=dict)
    # samplePointsFname: str = 'samplePoints.json'
    timeSeries: pd.DataFrame = field(default_factory=pd.DataFrame)
    timeSeriesFname: str = 'interpolatedTimeSeries.csv'
    neighbors: int = 20

    def __post_init__(self):
        super().__post_init__()
        self.getSamplePoints()
        self.getTimeSeries()        
        for variableName,vars in self.fileIndex.items():
            for year,filePath in vars.items():
                self.interpolateValues(year,variableName,filePath)
        self.timeSeries.to_csv(self.timeSeriesFname)
        

    def interpolateValues(self,year,variableName,filePath,kernel = 'thin_plate_spline'):

        index,xy,data = self.read(variableName,filePath)
        sites = [s for s in self.samplePoints.siteID if 
                 s in self.timeSeries.columns.get_level_values(0) and
                 variableName in self.timeSeries.columns.get_level_values(1)]
        summary = self.timeSeries.loc[self.timeSeries.index.year==year,(sites,variableName)].count()
        sites = summary.index[summary<index.shape].get_level_values(0)
        g = self.samplePoints.loc[sites].geometry
        targetPoints = np.array([g.x,g.y]).T.astype('float32')    
        # add units (missing from new entries)
        if len(sites):
            print(f'Interpolating {variableName}-{year} for: {sites}')
            
            interpolatedValues = pd.DataFrame(
                index=index,
                columns=pd.MultiIndex.from_arrays([sites,[variableName for s in sites],[self.metadata[variableName]['units'] for s in sites]]),
                data = [
                    RBFInterpolator(xy, data[i,:,:].flatten(), kernel=kernel,neighbors=self.neighbors)(targetPoints).astype('float32')
                    for i in range(index.shape[0])
                ])
            self.timeSeries.loc[interpolatedValues.index,interpolatedValues.columns] = interpolatedValues.copy()
    
    def getTimeSeries(self):
        index = pd.date_range(start=f'{min(self.years)}-01-01',end=f'{max(self.years)+1}-01-01',freq='3h',inclusive='left').tz_localize('UTC')
        self.timeSeriesFname = os.path.join(self.downloadPath,self.timeSeriesFname)
        c1,c2=np.meshgrid(self.samplePoints.siteID.values,self.variableNames)
        c1,c2=c1.flatten(),c2.flatten()
        c3 = [self.metadata[c]['units'] for c in c2]
        self.timeSeries = pd.DataFrame(index=index,columns=pd.MultiIndex.from_arrays([c1,c2,c3]))
        if os.path.isfile(self.timeSeriesFname):
            tx = pd.read_csv(self.timeSeriesFname,index_col=0,parse_dates=[0],header=[0,1,2])
            ix = tx.index.isin(self.timeSeries.index)
            cx = tx.columns.isin(self.timeSeries.columns)
            # Add the exiting row/column pairs within the current query
            self.timeSeries.loc[tx[ix].index,tx.columns[cx]] = tx.loc[tx[ix].index,tx.columns[cx]]
            # Add the missing columns within the query time-frame
            self.timeSeries[tx.columns[~cx]]=tx.loc[tx[ix].index,tx.columns[~cx]]
            # add the missing timeframe
            self.timeSeries = pd.concat([self.timeSeries,tx.loc[tx[~ix].index]]).sort_index()

    def getSamplePoints(self):
        if isinstance(self.samplePoints,str):
            with open(self.samplePoints) as f:
                self.samplePoints = yaml.safe_load(f)
            self.samplePoints = pd.DataFrame.from_dict(self.samplePoints,orient='index')
        self.samplePoints = gpd.GeoDataFrame(
            data=self.samplePoints, geometry=gpd.points_from_xy(self.samplePoints['longitude'], self.samplePoints['latitude']), crs="EPSG:4326"
        )
        self.samplePoints = self.samplePoints.to_crs(self.NARR_LCC)

@dataclass(kw_only=True)
class zonalStats(narrData):
    samplePolygons: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.samplePolygons,str):
            self.samplePolygons = gpd.read_file(self.samplePolygons)
        self.samplePolygons = self.samplePolygons.to_crs(self.NARR_LCC)