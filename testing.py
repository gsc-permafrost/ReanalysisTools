import os
import yaml
import scripts.NARR as NARR

# with open('siteID.yml') as f:
#     sampleSites = yaml.safe_load(f)

downloadPath = r'E:\data-dump\ncFiles'
if not os.path.isdir(downloadPath):
    downloadPath = r'D:\data-dump\ncFiles'
# NARR.interpolateNARR(level='monolevel',variableName='air.2m',date=2025,samplePoints={'name':['SCL'],'lat':[69],'lon':[-135]},grid_pad=0)
# dNarr = NARR.getNARR(
#     downloadPath=downloadPath,
#     level='monolevel',
#     variableNames=['dswrf'],#'prmsl ',
#     dates=[i for i in range(2024,2027)],
#     samplePoints={
#         'name':['SCL','BSP'],
#         'lat':[69.229178,69.319431,],
#         'lon':[-135.257278,-135.478286]})
# breakpoint()
px = NARR.pointEstimates(
    downloadPath=downloadPath,
    years=[i for i in range(2024,2027)],
    variableNames=['dswrf','dlwrf'],#,'uswrf','ulwrf','air.2m','apcp','pres','snod ','rhum','pblh','uwnd','vwnd'],
    samplePoints='siteID.yml')
