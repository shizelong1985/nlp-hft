import xml.etree.ElementTree as ET

filepath = r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\DJG-US-20200323-20200330\test.nml'
tree = ET.parse(filepath)
root = tree.getroot()
