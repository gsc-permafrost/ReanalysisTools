import os
import yaml
import scripts.NARR as NARR

downloadPath = r'E:\data-dump\ncFiles'
if not os.path.isdir(downloadPath):
    downloadPath = r'D:\data-dump\ncFiles'

px = NARR.pointEstimates(
    downloadPath=downloadPath,
    years=[i for i in range(2024,2027)],
    variableNames=['dswrf','dlwrf','uswrf.sfc','ulwrf.sfc','air.2m'],#,'apcp','pres','snod ','rhum','pblh','uwnd','vwnd'],
    samplePoints='siteID.yml')

breakpoint()