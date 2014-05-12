import pylab as pl
import numpy as np
from XmlAnnotationReader import XmlAnnotationReader
from DicomFolderReader import DicomFolderReader

for setID in range(41,51):
    try:
        dfr = DicomFolderReader.create("../data/LIDC-IDRI", setID)
    except:
        continue
    
    reader = dfr.getAnnotationReader()
    print setID, len(reader.Nodules)

# numbS = []
# 
# for myPath in DicomFolderReader.findAllPaths("../data/LIDC-IDRI"):
#         dfr = DicomFolderReader(myPath, False)
#         aSlices = dfr.getVolumeShape()
#         #print(aSlices)
#         numb = aSlices[2]
#         #print(numb)
#         numbS.append(numb)
# #print(numbS)
# #print(len(numbS))
# 
# gem = numbS[0:30]
# print(gem)
# a=sum(gem)
# print(a)
# gem = sum(gem)/len(gem)
# print(gem)
# 
# 
# 
# gemV = numbS[30:]
# print(gemV)
# b = sum(gemV)/1
# print(b)
# gemV = sum(gemV)/len(gemV)
# print(gemV)



        
        