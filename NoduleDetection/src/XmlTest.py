from lxml import etree
from Nodule import *

tree = etree.parse("../data/069.xml")
root = tree.getroot()

#print(etree.tostring(root, pretty_print=True))

readingSessions = root.findall("{http://www.nih.gov}readingSession")
nbSessions = len(readingSessions)
print("XML contains {0} reading sessions.".format(nbSessions))

if nbSessions > 0:
    firstSession = readingSessions[0]
    noduleNodes = firstSession.findall("{http://www.nih.gov}unblindedReadNodule")
    nbNoduleNodes = len(noduleNodes)
    print("First session contains {0} nodules:".format(nbNoduleNodes))
    for noduleNode in noduleNodes:
        nodule = Nodule.fromXML(noduleNode)
        #print("\tSubtlety={0}".format(nodule.subtlety))
        print("\tFound {0} regions for nodule {1}.".format(nodule.getNbRegions(), nodule.ID))
        for worldZ in sorted(nodule.regions.iterkeys()):
            coords = nodule.getRegionCoords(worldZ)
            nbCoords = len(coords)
            print("\t\tFound {0} coordinates for region with worldZ={1}:".format(nbCoords, worldZ))
            for coord in coords:
                print("\t\t\t{0}".format(coord))
    