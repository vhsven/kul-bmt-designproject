import numpy as np
from lxml import etree
from Nodule import Nodule
#from DicomFolderReader import DicomFolderReader 
from os import listdir
from os.path import isfile, join

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
            nodules.append(nodule)
            
        return nodules 
    
    def getNodulePositions(self): #in pixel coordinates
        for nodule in self.Nodules:
            regionCenters, regionRs = nodule.Regions.getRegionCenters()
            for pixelZ in regionCenters.keys():                
                yield np.append(regionCenters[pixelZ], pixelZ), regionRs[pixelZ]
                
    def getNodulePositionsInSlice(self, mySlice):
        for nodule in self.Nodules:
            regionCenters, regionRs = nodule.Regions.getRegionCenters()
            for pixelZ in regionCenters.keys():
                if int(pixelZ) == int(mySlice):
                    yield regionCenters[pixelZ], regionRs[pixelZ]
            