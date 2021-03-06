import numpy as np
from lxml import etree
from Nodule import Nodule
#from DicomFolderReader import DicomFolderReader 
from os import listdir
from os.path import isfile, join
from Constants import IGNORE_ONE_PIXEL_NODULES

class XmlAnnotationReader:
    def __init__(self, myPath, cc):
        myFiles = [ join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f)) and f.lower().endswith(".xml") ]
        assert len(myFiles) == 1 #there must be only 1 XML file is this directory
        myFile = myFiles[0]
        #self.dfr = DicomFolderReader(myPath)
        #cc = self.dfr.getCoordinateConverter()
        tree = etree.parse(myFile)
        root = tree.getroot()
        self.Nodules = self.parseRoot(root, cc)
    
    def __del__(self):
        del self.Nodules
        #del self.dfr
        
    def __str__(self):
        return "XmlAnnotationReader with {} nodules.".format(len(self.Nodules))
        
    def parseRoot(self, rootNode, cc):
        #print(etree.tostring(rootNode, pretty_print=True))
        
        readingSessions = rootNode.findall("{http://www.nih.gov}readingSession")
        nbSessions = len(readingSessions)
        #print("XML contains {0} reading sessions.".format(nbSessions))

        assert nbSessions > 0
        firstSession = readingSessions[0]
        return self.parseReadingSession(firstSession, cc)
                                  
    def parseReadingSession(self, sessionNode, cc):
        noduleNodes = sessionNode.findall("{http://www.nih.gov}unblindedReadNodule")
        nodules = []
        #print("First session contains {0} nodules:".format(len(noduleNodes)))
        for noduleNode in noduleNodes:
            nodule = Nodule.fromXML(noduleNode, cc)
            if not (IGNORE_ONE_PIXEL_NODULES and nodule.OnePixel):
                nodules.append(nodule)
            
        return nodules 
    
    def getNodulePositions(self, minOrMax): #in pixel coordinates
        for nodule in self.Nodules:
            regionCenters, regionRs = nodule.Regions.getRegionCenters(minOrMax)
            for pixelZ in nodule.Regions.getSortedZIndices():                
                yield np.append(regionCenters[pixelZ], pixelZ), regionRs[pixelZ]
                
    def getNodulePositionsInSlice(self, mySlice, minOrMax):
        for nodule in self.Nodules:
            regionCenters, regionRs = nodule.Regions.getRegionCenters(minOrMax)
            for pixelZ in nodule.Regions.getSortedZIndices():
                if int(pixelZ) == int(mySlice):
                    yield regionCenters[pixelZ], regionRs[pixelZ]
                    
    def getNodulesMask(self, shape, minOrMax, radiusFactor=1.0):
        mask3D = np.zeros(shape, dtype=np.bool)
        for nodule in self.Nodules:
            np.bitwise_or(mask3D, nodule.Regions.getRegionMask3D(shape, minOrMax, radiusFactor), out=mask3D)
            
        return mask3D
            