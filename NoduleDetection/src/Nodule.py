from NoduleRegions import NoduleRegions

class Nodule:
    def __init__(self, noduleID):
        self.ID = noduleID
        self.subtlety = -1
        self.internalStructure = -1
        self.calcification = -1
        self.sphericity = -1
        self.margin = -1
        self.lobulation = -1
        self.spiculation = -1
        self.texture = -1
        self.malignancy = -1
        self.regions = NoduleRegions()
                
    @staticmethod
    def fromXML(xml, cc):
        noduleID = xml.find("{http://www.nih.gov}noduleID").text
        chars = xml.find("{http://www.nih.gov}characteristics")
        nodule = Nodule(noduleID)
        
        #parse characterists
        if chars is not None:
            nodule.subtlety = int(chars.find("{http://www.nih.gov}subtlety").text)
            nodule.internalStructure = int(chars.find("{http://www.nih.gov}internalStructure").text)
            nodule.calcification = int(chars.find("{http://www.nih.gov}calcification").text)
            nodule.sphericity = int(chars.find("{http://www.nih.gov}sphericity").text)
            nodule.margin = int(chars.find("{http://www.nih.gov}margin").text)
            nodule.lobulation = int(chars.find("{http://www.nih.gov}lobulation").text)
            nodule.spiculation = int(chars.find("{http://www.nih.gov}spiculation").text)
            nodule.texture = int(chars.find("{http://www.nih.gov}texture").text)
            nodule.malignancy = int(chars.find("{http://www.nih.gov}malignancy").text)
        
        #parse regions of interest    
        regionList = xml.findall("{http://www.nih.gov}roi")
        nbRegions = len(regionList)
        if nbRegions > 0:
            for roi in regionList:
                worldZ = float(roi.find("{http://www.nih.gov}imageZposition").text)
                pixelZ = cc.getPixelZ(worldZ)
                #pizelZ = int(round(pizelZ))
                edgeMapList = roi.findall("{http://www.nih.gov}edgeMap")
                coordList = []
                for edgeMap in edgeMapList:
                    voxelX = int(edgeMap.find("{http://www.nih.gov}xCoord").text)
                    voxelY = int(edgeMap.find("{http://www.nih.gov}yCoord").text)
                    coord = voxelX, voxelY, pixelZ #tuple
                    coordList.append(coord)
                    
                nodule.regions.addRegion(pixelZ, coordList)
        
        return nodule