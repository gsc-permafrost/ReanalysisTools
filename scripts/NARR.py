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

    def __post_init__(self):
        self.findFiles()

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
                filePath,flag=self.timeCheck(filePath)
        elif exists and subFolder == 'partialYear':
            filePath,flag=self.timeCheck(filePath)
            if flag:
                filePath = self.getFile(year,variableName,subFolder='partialYear')
        return filePath
    
    def timeCheck(self,filePath):
        with netCDF4.Dataset(filePath) as dataset:
            tx = self.getTime(dataset)
        if tx.month.max()==12 and 'partialYear' in filePath:
            fp = filePath.replace('partialYear','fullYear')
            if not os.path.isdir(os.path.split(fp)[0]):
                os.makedirs(os.path.split(fp)[0])
            shutil.move(filePath,fp)
            filePath = fp
            flag = False
        else:
            if tx.month.max() != datetime.datetime.now().month-1 and (time.time()-os.path.getctime(filePath))/(3660*24) >1:
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
        with netCDF4.Dataset(filePath) as dataset:
            tx = self.getTime(dataset)
            lon = np.ma.getdata(dataset.variables['lon'][:])
            lat = np.ma.getdata(dataset.variables['lat'][:])
            x = np.ma.getdata(dataset.variables['x'][:])
            y = np.ma.getdata(dataset.variables['y'][:])
            if 'variableName' not in self.metadata:
                self.metadata[variableName] = dataset.variables[variableName.split('.')[0]].__dict__
            data = np.ma.getdata(dataset.variables[variableName.split('.')[0]][:])
            
            x,y = np.meshgrid(x,y)
            x,y = x.flatten(),y.flatten()
            xy = np.array([x,y]).T
        return(tx,xy,data)


@dataclass(kw_only=True)
class pointEstimates(narrData):
    samplePoints: gpd.GeoDataFrame = field(default_factory=dict)
    samplePointsFname: str = 'samplePoints.json'
    timeSeries: pd.DataFrame = field(default_factory=pd.DataFrame)
    timeSeriesFname: str = 'interpolatedTimeSeries.csv'
    neighbors: int = 20

    def __post_init__(self):
        self.getSamplePoints()
        self.getTimeSeries()
        super().__post_init__()        
        for variableName,vars in self.fileIndex.items():
            for year,filePath in vars.items():
                self.interpolateValues(year,variableName,filePath)
        self.timeSeries.to_csv(self.timeSeriesFname)
        mdF = os.path.join(self.downloadPath,'metadata.yml')
        if os.path.isfile(mdF):
            with open(mdF) as f:
                mdIn = yaml.safe_load(f)
        else:
            mdIn = {}
        cleanMD = {key:{k:v.item() if isinstance(v,np.generic) else v if not isinstance(v,np.ndarray) else v.tolist() for k,v in value.items()} for key,value in self.metadata.items()}
        mdOut = mdIn | cleanMD
        with open(mdF,'w+') as f:
            yaml.safe_dump(mdOut,f)
        

    def interpolateValues(self,year,variableName,filePath,kernel = 'thin_plate_spline'):

        index,xy,data = self.read(variableName,filePath)
        summary = self.timeSeries.loc[self.timeSeries.index.year==year,variableName].count()
        sites = summary.index[summary<index.shape]
        sites = [v for v in sites[sites.isin(self.samplePoints.siteID)]]
        g = self.samplePoints.loc[sites].geometry
        targetPoints = np.array([g.x,g.y]).T.astype('float32')       
        if len(sites):
            print(f'Interpolating {variableName}-{year} for: {sites}')
            
            interpolatedValues = pd.DataFrame(
                index=index,
                columns=pd.MultiIndex.from_arrays([[variableName for s in sites],sites]),
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
        self.timeSeries = pd.DataFrame(index=index,columns=pd.MultiIndex.from_arrays([c2,c1]))
        if os.path.isfile(self.timeSeriesFname):
            tx = pd.read_csv(self.timeSeriesFname,index_col=0,parse_dates=[0],header=[0,1])
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
        # WKT description of the NARR LCC projection, source: https://spatialreference.org/ref/sr-org/8214/
        NARR_LCC = '+proj=lcc +lat_1=50 +lat_0=50 +lon_0=-107 +k_0=1 +x_0=5632642.22547 +y_0=4612545.65137 +a=6371200 +b=6371200 +units=m +no_defs'
        self.samplePoints = self.samplePoints.to_crs(NARR_LCC)

@dataclass(kw_only=True)
class getNARR:
    variableNames: list
    dates: list
    level: str

    # samplePoints: dict = field(default_factory=lambda:{
    #     'siteID':[],'lat':[],'lon':[]
    # })
    neighbors: int = 20
    metadata: dict = field(init=False,default_factory=dict)
    # timeSeries: dict = field(init=False,default_factory=dict)

    def __post_init__(self):
        self.setTarget()
        self.getData()

    def setTarget(self):
        if not isinstance(self.samplePoints,gpd.GeoDataFrame):
            self.samplePoints = gpd.GeoDataFrame(
                data=self.samplePoints, geometry=gpd.points_from_xy(self.samplePoints['lon'], self.samplePoints['lat']), crs="EPSG:4326"
            )
             
        # WKT description of the NARR LCC projection
        # Source: https://spatialreference.org/ref/sr-org/8214/
        NARR_LCC = '+proj=lcc +lat_1=50 +lat_0=50 +lon_0=-107 +k_0=1 +x_0=5632642.22547 +y_0=4612545.65137 +a=6371200 +b=6371200 +units=m +no_defs'
        self.samplePoints = self.samplePoints.to_crs(NARR_LCC)
        
        self.target = np.array([self.samplePoints.geometry.x,self.samplePoints.geometry.y]).T.astype('float32')

    def getData(self):
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
        for year,vars in self.fileIndex.items():
            self.timeSeries[year] = {}
            for vn,fn in vars.items():
                fp,exits = self.filePath(fn)
                if not exits:
                    url = self.urlPath(fn)
                    print(f'downloading: {url}')
                    if not os.path.isdir(self.downloadPath):
                        os.makedirs(self.downloadPath)
                    urllib.request.urlretrieve(url,fp)
                    print(f'saved: {fp}')
                self.timeSeries[year][vn] = self.readData(year,vn,fp)
            self.timeSeries[year] = pd.concat([self.timeSeries[year][vn] for vn in vars.keys()],axis=1)
        self.timeSeries = pd.concat([self.timeSeries[year] for year in self.timeSeries.keys()],axis=0)
        self.timeSeries.to_csv(os.path.join(self.downloadPath,'interpolatedTimeSeries.csv'))


    def filePath(self,fn):
        fp = os.path.join(self.downloadPath,fn)
        return(fp,os.path.isfile(fp))
    
    def urlPath(self,fn):
        return(f"{self.baseURL}/{self.level}/{fn}")

    def readData(self,year,vn,fp):
        dataset = netCDF4.Dataset(fp)
        tx = dataset.variables['time']
        tx = netCDF4.num2date(tx[:], tx.units,calendar = 'standard',only_use_cftime_datetimes=False)
        index = pd.to_datetime(tx).tz_localize('UTC')
        self.lon = np.ma.getdata(dataset.variables['lon'][:])
        self.lat = np.ma.getdata(dataset.variables['lat'][:])
        self.x = np.ma.getdata(dataset.variables['x'][:])
        self.y = np.ma.getdata(dataset.variables['y'][:])

        self.metadata[vn] = dataset.variables[vn.split('.')[0]].__dict__
        data = np.ma.getdata(dataset.variables[vn.split('.')[0]][:])

        x,y = np.meshgrid(self.x,self.y)
        x,y = x.flatten(),y.flatten()
        self.xy = np.array([x,y]).T

        print(f'Interpolating {vn}-{year}: ')
        T1 = time.time()
        interpolatedValues = np.array(
                    [self.interpolate(data[i,:,:].flatten()) for i in range(index.shape[0])]
                )
        print('runtime = ',round(time.time()-T1,2))
        interpolatedValues[np.where(interpolatedValues<self.metadata[vn]['actual_range'].min())]=self.metadata[vn]['actual_range'].min()
        interpolatedValues[np.where(interpolatedValues>self.metadata[vn]['actual_range'].max())]=self.metadata[vn]['actual_range'].max()
        interpolatedValues = pd.DataFrame(
            index = index,
            columns=[f"{vn}_{n}" for n in self.samplePoints['siteID']],
            data = interpolatedValues)
        return(interpolatedValues)

    def interpolate(self,value,kernel='thin_plate_spline'):
        # Interpolates value from grid (xy) to desired points (coords) using a Radial Bias Function
        # Default behavior is to use a thin plate spline function r**2 * log(r)

        return(
            RBFInterpolator(self.xy, value, kernel=kernel,neighbors=self.neighbors)(self.target).astype('float32')
            )

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
                    self.metadata[vn] = dataset.variables[vn.split('.')[0]].__dict__
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
        'siteID':[],'lat':[],'lon':[]
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

        gridPoints = gpd.GeoDataFrame(index = np.arange(0,self.lon.shape[0]*self.lon.shape[1]),geometry=gpd.points_from_xy(x,y))
        gridIndex = gridPoints.index.values.reshape(self.lat.shape)
        selection = gridPoints[gridPoints.within(searchBuffer)]
        key = np.where(np.isin(gridIndex,selection.index.values))

        # xi,yi = np.meshgrid(selection.geometry.x.values,selection.geometry.y.values)
        # xi,yi = xi.flatten(),yi.flatten()
        self.xy = np.array([selection.geometry.x.values,selection.geometry.y.values]).T

        self.dataSelection = {var:self.data[var][:,key[0],key[1]] for var in self.variableNames}

        self.timeSeries = pd.concat(
            [pd.DataFrame(
                index=self.index,
                columns=[f"{vn}_{n}" for n in self.samplePoints['siteID']],
                data = np.array(
                    [self.interpolate(self.data[vn][i,key[0],key[1]]) for i in range(self.index.shape[0])]
                )
            )
            for vn in self.variableNames],axis=1
        )

        # Make plot of grid cells
        # lon_box = self.lon[key[0],key[1]]
        # lat_box = self.lat[key[0],key[1]]

        # m = folium.Map(location=[self.samplePoints.lat[0],self.samplePoints.lon[0]])   
        # for at,on in zip (lat_box.flatten(),lon_box.flatten()):
        #     folium.Marker([at, on]).add_to(m)
        # folium.CircleMarker([self.samplePoints.lat[0],self.samplePoints.lon[0]],popup=self.samplePoints.siteID[0]).add_to(m)
        # m.save(os.path.join(self.downloadPath,f'{self.samplePoints.siteID[0]}_{self.variableNames}_grid_pts.html'))

    def interpolate(self,value,kernel='linear'):
        # Interpolates value from grid (xy) to desired points (coords) using a Radial Bias Function
        # Default behavior is to use a thin plate spline function r**2 * log(r)

        vx = RBFInterpolator(self.xy, value, kernel=kernel)(self.target)
        # if not self.extrapolate:
        #     vx[vx<value.min()]=value.min()
        #     vx[vx>value.max()]=value.max()
        return(vx)
