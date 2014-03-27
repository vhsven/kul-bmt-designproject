from lxml import etree
from Nodule import *
from CoordinateConverter import CoordinateConverter
from DicomFolderReader import DicomFolderReader 
from os import listdir
from os.path import isfile, join

class XmlAnnotationReader:
    def __init__(self, myPath):
        myFiles = [ join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f)) and f.lower().endswith(".xml") ]
        assert len(myFiles) == 1 #there must be only 1 XML file is this directory
        myFile = myFiles[0]
        dfr = DicomFolderReader(myPath)
        matrix = dfr.getWorldMatrix()
        cc = CoordinateConverter(matrix)
        tree = etree.parse(myFile)
        root = tree.getroot()
        self.Nodules = self.parseRoot(root, cc)
        
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
                        
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
for nodule in reader.Nodules:
    nodule.printRegions()