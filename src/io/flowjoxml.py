'''
Created on Feb 11, 2012

@author: Jacob Frelinger
'''
import xml.etree.ElementTree as xml
import sys
import numpy
from fcm import PolyGate

class FlowjoWorkspace(object):
    '''
    Doccument to handle opening flowjow workspaces
    '''


    def __init__(self,gates, comp=None):
        '''
        gates - dictionary of gates keyed by filename,
        comp - dictionary of defined compensation channels and matricies, keyed by name defaults to none
        '''
        self.gates = gates
        self.comp = comp
        
    @property
    def files(self):
        return self.gates.keys()
    

def load_flowjo_xml(fh):
    '''
    create a FlowjoWorkspace object from a xml file
    '''
    if isinstance(fh,str):
        fh = open(fh,'r')
    
    tree = xml.parse(fh)
    print tree
    root = tree.getroot()
    files = {}
    chans = []
    for node in root.iter('SampleNode'):
        files[node.attrib['name']] = {}
        for gate in node.iter('PolygonGate'):
            rect = gate.find('PolyRect')
            if rect is not None:
                verts = []
                axis = (rect.attrib['xAxisName'], rect.attrib['yAxisName'])
                for vert in rect.iter('Vertex'):
                    verts.append((float(vert.attrib['x']), float(vert.attrib['y'])))
                files[node.attrib['name']][rect.attrib['name']] = PolyGate(verts, axis, rect.attrib['name'])
            poly = gate.find('Polygon')
            if poly is not None:
                verts = []
                axis = (poly.attrib['xAxisName'], poly.attrib['yAxisName'])
                for vert in poly.iter('Vertex'):
                    verts.append((float(vert.attrib['x']), float(vert.attrib['y'])))
                files[node.attrib['name']][poly.attrib['name']] = PolyGate(verts, axis, poly.attrib['name'])
            
            elip = gate.find('Ellipse')
            if elip is not None:
                pass # TODO implement ellipse gate
#                verts = []
#                axis = (elip.attrib['xAxisName'], elip.attrib['yAxisName'])
#                for vert in poly.iter('Vertex'):
#                    verts.append(float(vert.attrib['x']), float(vert.attrib['y']))
#                files[node.attrib['name']][elip.attrib['name']] = PolyGate(verts, axis, rect.attrib['name'])
                            
    comps = {}
    for mats in root.iter('CompensationMatrices'):
        for mat in mats.iter('CompensationMatrix'):
            a = len([ i for i in mat.iter('Channel')])
            comp = numpy.zeros((a,a))
            chans = []
            for i,chan in enumerate(mat.iter('Channel')):
                chans.append(chan.attrib['name'])
                for j,sub in enumerate(chan.iter('ChannelValue')):
                    comp[i,j] = float(sub.attrib['value'])
            comps[mat.attrib['name']] = (chans,comp)
    if len(comps) > 0:
        return FlowjoWorkspace(files,comps)
    else:
        return FlowjoWorkspace(files)
    
    
    
if __name__ == "__main__":
    a = FlowjoWorkspace({"a":0})
    print a.files
    try:
        a.files = 'bar'
    except AttributeError:
        pass
    
    a = load_flowjo_xml('/home/jolly/Projects/fcm/scratch/flowjoxml/pretty.xml')
    print a.files
    print a.gates['Specimen_001_A1_A01.fcs']
    print a.comp.keys(), a.comp['Comp Matrix']