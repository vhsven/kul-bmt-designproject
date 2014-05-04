from DicomFolderReader import DicomFolderReader

for myPath in DicomFolderReader.findPaths("../data/LIDC-IDRI"):
    dfr = DicomFolderReader(myPath)
    dfr.compress()
    
    #data = dfr.getVolumeData()
    #print dfr.getVoxelShape(), myPath
    
    #data = dfr.getVolumeData()
    #print data.min(), data.max()
    
    #print dfr.Slices[0]