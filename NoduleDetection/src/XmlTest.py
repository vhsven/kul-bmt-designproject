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
        print("\tID={0}".format(nodule.ID))
        print("\tSubtlety={0}".format(nodule.subtlety))
    