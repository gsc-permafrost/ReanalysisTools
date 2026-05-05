import scripts.NARR as NARR

# NARR.interpolateNARR(level='monolevel',variableName='air.2m',date=2025,samplePoints={'name':['SCL'],'lat':[69],'lon':[-135]},grid_pad=0)
dNarr = NARR.interpolateNARR(level='monolevel',variableNames=['prmsl ','air.2m','dswrf'],dates=[i for i in range(2024,2027)],samplePoints={'name':['SCL','BSP'],'lat':[69.229178,69.319431,],'lon':[-135.257278,-135.478286]},grid_pad=0)
breakpoint()