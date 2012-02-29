'''
Created on Feb 11, 2012

@author: Jacob Frelinger
'''
import xml.etree.ElementTree as xml
import sys
import numpy
from fcm import PolyGate


class PopulationNode(object):
    '''
    node for gates in xml tree
    '''
    def __init__(self, name, gate):
        self.name = name
        self.gate = gate
        self.subpops = {}
        
    @property
    def gates(self):
        a = [self.gate]
        if self.subpops:
            for i in self.subpops:
                a.extend(self.subpops[i].gates)
        return a
    

    def pprint(self, depth=0):
        j = "  " * depth + self.name + "\n"
        if self.subpops is not None:
            for i in self.subpops:
                j += self.subpops[i].pprint(depth+1)
            
        return j    
class xmlfcsfile(object):
    '''
    container object for fcs file defined in a flowjo xml file
    '''
    def __init__(self, name, comp=None, pops = None):
        self.name = name
        if pops is not None:
            self.pops = {}
        else:
            self.pops = pops
        self.comp = comp
    
    @property
    def gates(self):
        a = []
        for i in self.pops:
            a.append(self.pops[i].gates)
        return a
    
    def pprint(self, depth=0):
        j = "  " * depth + self.name + "\n"
        for i in self.pops:
            j += self.pops[i].pprint(depth+1)
            
        return j
    
    
class FlowjoWorkspace(object):
    '''
    Object representing the files, gates, and compensation matricies from a 
    flowjo xml worksapce
    '''


    def __init__(self,tubes, comp=None):
        '''
        gates - dictionary of gates keyed by filename,
        comp - dictionary of defined compensation channels and matricies, keyed by name defaults to none
        '''
        
        self.tubes = tubes
        self.comp = comp
        
    @property
    def file_names(self):
        return [self.tubes[tube].name for tube in self.tubes]
    
    @property
    def gates(self):
        gates = {}
        for tube_name in self.tubes:
            tube = self.tubes[tube_name]
            gates[tube_name] = tube.gates
        return gates

    def pprint(self):
        j = ''
        for i in self.tubes:
            j += self.tubes[i].pprint(0)
        
        return j
            
def load_flowjo_xml(fh):
    '''
    create a FlowjoWorkspace object from a xml file
    '''
    if isinstance(fh,str):
        fh = open(fh,'r')
    
    tree = xml.parse(fh)
    
    root = tree.getroot()
    files = {}
    chans = []
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
    
    for node in root.iter('Sample'):
        # pull out comp matrix
        keywords = node.find('Keywords')
        comp_matrix = ''
        if keywords is not None:
            
            for keyword in keywords.iter('Keyword'):
                if 'name' in keyword.attrib:
                    if keyword.attrib['name'] == 'FJ_CompMatrixName':
                        comp_matrix = comps[keyword.attrib['value']] 
        sample = node.find('SampleNode')
        if comp_matrix: 
            fcsfile = xmlfcsfile(sample.attrib['name'], comp = comp_matrix)
        else:
            fcsfile = xmlfcsfile(sample.attrib['name'])                      
        # find gates
        fcsfile.pops = find_pops(node)
        files[fcsfile.name] = fcsfile
                            
    if len(comps) > 0:
        return FlowjoWorkspace(files,comps)
    else:
        return FlowjoWorkspace(files)
 
 
def find_pops(node):
    subpops= {}
    for pop in node.iter('Population'):
        
        for popgate in pop.iter('PolygonGate'):
            rect = popgate.find('PolyRect')
            if rect is not None:
                verts = []
                axis = (rect.attrib['xAxisName'], rect.attrib['yAxisName'])
                for vert in rect.iter('Vertex'):
                    verts.append((float(vert.attrib['x']), float(vert.attrib['y'])))
                gate = PolyGate(verts, axis, rect.attrib['name'])
                subpop = PopulationNode(rect.attrib['name'], gate)
                a = pop.find('Population')
                if a is not None:

                    subpop.subgates = find_pops(a)
                subpops[rect.attrib['name']] = subpop
            poly = popgate.find('Polygon')
            if poly is not None:
                verts = []
                axis = (poly.attrib['xAxisName'], poly.attrib['yAxisName'])
                for vert in poly.iter('Vertex'):
                    verts.append((float(vert.attrib['x']), float(vert.attrib['y'])))
                gate = PolyGate(verts, axis, poly.attrib['name'])
                subpop = PopulationNode(poly.attrib['name'], gate)
                a = pop.find('Population')
                if a is not None:
                    subpop.subgates = find_pops(a)
                subpops[poly.attrib['name']] = subpop
            elip = popgate.find('Ellipse')
            if elip is not None:
                pass # TODO implement ellipse gate
#                verts = []
#                axis = (elip.attrib['xAxisName'], elip.attrib['yAxisName'])
#                for vert in poly.iter('Vertex'):
#                    verts.append(float(vert.attrib['x']), float(vert.attrib['y']))
#                files[node.attrib['name']][elip.attrib['name']] = PolyGate(verts, axis, rect.attrib['name'])   
    if subpops:
        return subpops
    else:
        return None
    
if __name__ == "__main__":
#    a = FlowjoWorkspace({"a":0})
#    print a.files
#    try:
#        a.files = 'bar'
#    except AttributeError:
#        pass
    
    a = load_flowjo_xml('/home/jolly/Projects/fcm/scratch/flowjoxml/pretty.xml')
    print a.file_names
    print a.gates
    print a.comp.keys(), '\n', a.comp['Comp Matrix']
    print a.tubes['Specimen_001_A1_A01.fcs'].comp[1] - a.comp['Comp Matrix'][1]
    print a.pprint()