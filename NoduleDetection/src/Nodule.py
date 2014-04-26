from NoduleRegions import NoduleRegions

class Nodule:
    def __init__(self, noduleID):
        self.ID = noduleID
        self.Subtlety = -1
        self.InternalStructure = -1
        self.Calcification = -1
        self.Sphericity = -1
        self.Margin = -1
        self.Lobulation = -1
        self.Spiculation = -1
        self.Texture = -1
        self.Malignancy = -1
        self.Regions = NoduleRegions()
    
    def __del__(self):
        del self.Regions
        
    @staticmethod
    def fromXML(xml, cc):
        noduleID = xml.find("{http://www.nih.gov}noduleID").text
        chars = xml.find("{http://www.nih.gov}characteristics")
        nodule = Nodule(noduleID)
        
        #parse characterists
        if chars is not None:
            nodule.Subtlety = int(chars.find("{http://www.nih.gov}subtlety").text)
            nodule.InternalStructure = int(chars.find("{http://www.nih.gov}internalStructure").text)
            nodule.Calcification = int(chars.find("{http://www.nih.gov}calcification").text)
            nodule.Sphericity = int(chars.find("{http://www.nih.gov}sphericity").text)
            nodule.Margin = int(chars.find("{http://www.nih.gov}margin").text)
            nodule.Lobulation = int(chars.find("{http://www.nih.gov}lobulation").text)
            nodule.Spiculation = int(chars.find("{http://www.nih.gov}spiculation").text)
            nodule.Texture = int(chars.find("{http://www.nih.gov}texture").text)
            nodule.Malignancy = int(chars.find("{http://www.nih.gov}malignancy").text)
        
        #parse regions of interest    
        regionList = xml.findall("{http://www.nih.gov}roi")
        nbRegions = len(regionList)
        if nbRegions > 0:
            for roi in regionList:
                worldZ = float(roi.find("{http://www.nih.gov}imageZposition").text)
                z = cc.getPixelZ(worldZ)
                #pizelZ = int(round(pizelZ))
                edgeMapList = roi.findall("{http://www.nih.gov}edgeMap")
                coordList = []
                for edgeMap in edgeMapList:
                    voxelX = int(edgeMap.find("{http://www.nih.gov}xCoord").text)
                    voxelY = int(edgeMap.find("{http://www.nih.gov}yCoord").text)
                    coord = voxelX, voxelY, z #tuple
                    coordList.append(coord)
                    
                nodule.Regions.addRegion(z, coordList)
        
        return nodule