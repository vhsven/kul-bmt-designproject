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
        #TODO add ROIs
      
    @staticmethod
    def fromXML(xml):
        noduleID = xml.find("{http://www.nih.gov}noduleID").text
        chars = xml.find("{http://www.nih.gov}characteristics")
        nodule = Nodule(noduleID)
        
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
            
        return nodule