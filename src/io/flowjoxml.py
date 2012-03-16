'''
Created on Feb 11, 2012

@author: Jacob Frelinger
'''

import xml.etree.cElementTree as xml
import numpy
from fcm import PolyGate


class PopulationNode(object):
    '''
    node for gates in xml tree
    '''
    def __init__(self, name, gate, subpops = None):
        self.name = name
        self.gate = gate
        if subpops is None:
            self.subpops = {}
        else:
            self.subpops = subpops
        
    @property
    def gates(self):
        a = [self.gate]
        for i in self.subpops:
            a.extend([self.subpops[i].gates])
        if len(a) == 1:
            return a[0]
        else:
            return a
    

    def pprint(self, depth=0):
        j = "  " * depth + self.name + "\n"
        for i in self.subpops:
            j += self.subpops[i].pprint(depth+1)            
        return j    
    
    def apply_gates(self, file):
        self.gate.gate(file)
        node = file.current_node
        for i in self.subpops:
            file.visit(node)
            self.subpops[i].apply_gates(file)
        
    
class xmlfcsfile(object):
    '''
    container object for fcs file defined in a flowjo xml file
    '''
    def __init__(self, name, comp=None, pops = None):
        self.name = name
        if pops is None:
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
        if self.pops is not None:
            for i in self.pops:
                j += self.pops[i].pprint(depth+1)
            
        return j
    
    def apply_gates(self, file):
        node = file.current_node
        for i in self.pops:
            file.visit(node)
            self.pops[i].apply_gates(file)
    
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
    prefix = ''
    suffix = ''
    psdict = {}
    for mats in root.iter('CompensationMatrices'):
        for mat in mats.iter('CompensationMatrix'):
            prefix = mat.attrib['prefix']
            suffix = mat.attrib['suffix']
            
            a = len([ i for i in mat.iter('Channel')])
            comp = numpy.zeros((a,a))
            chans = []
            for i,chan in enumerate(mat.iter('Channel')):
                chans.append(chan.attrib['name'])
                for j,sub in enumerate(chan.iter('ChannelValue')):
                    comp[i,j] = float(sub.attrib['value'])
            comps[mat.attrib['name']] = (chans,comp)
            psdict[mat.attrib['name']] = (prefix, suffix)
    
    for node in root.iter('Sample'):
        # pull out comp matrix
        keywords = node.find('Keywords')
        comp_matrix = ''
        if keywords is not None:
            
            for keyword in keywords.iter('Keyword'):
                if 'name' in keyword.attrib:
                    if keyword.attrib['name'] == 'FJ_CompMatrixName':
                        if keyword.attrib['value'] in comps:
                            comp_matrix = comps[keyword.attrib['value']]
                            
                        else:
                            comp_matrix = None
                        if keyword.attrib['value'] in psdict:
                            prefix, suffix = psdict[keyword.attrib['value']]
        sample = node.find('SampleNode')
        if comp_matrix: 
            fcsfile = xmlfcsfile(sample.attrib['name'], comp = comp_matrix)
        else:
            fcsfile = xmlfcsfile(sample.attrib['name'])                      
        # find gates
        fcsfile.pops = find_pops(sample, prefix, suffix)
        files[fcsfile.name] = fcsfile
                            
    if len(comps) > 0:
        return FlowjoWorkspace(files,comps)
    else:
        return FlowjoWorkspace(files)
 
 
def find_pops(node, prefix=None, suffix=None):
    pops = {}
    for i in node:
        if i.tag == 'Population':
            pops[i.attrib['name']] = build_pops(i, prefix, suffix)
    return pops
    
    
def build_pops(node,prefix=None, suffix=None):
    #print "*"*depth, node.tag, node.attrib['name']
    name = node.attrib['name']
    
    children = {}
    for i in node:
        if i.tag == 'Population':
            tmp = build_pops(i, prefix, suffix)
            if tmp is not None:
                children[i.attrib['name']] = tmp
            
        elif i.tag == 'PolygonGate':
            for j in i:
                if j.tag == 'PolyRect':
                    g = build_Polygon(j, prefix, suffix)
                elif j.tag == 'Polygon':
                    g = build_Polygon(j, prefix, suffix)
    try:
        return  PopulationNode(name, g, children)
    except UnboundLocalError:
        return None
    

def build_Polygon(rect, prefix=None, suffix=None):
    verts = []
    axis = [rect.attrib['xAxisName'], rect.attrib['yAxisName']]
    if prefix is not None:
        for i,j in enumerate(axis):
            if j.startswith(prefix):
                axis[i] = j.replace(prefix,'')
    if suffix is not None:
        for i,j in enumerate(axis):
            if j.endswith(suffix):
                axis[i] = j[:-(len(suffix))]
    axis = tuple(axis)
    for vert in rect.iter('Vertex'):
        verts.append((float(vert.attrib['x']), float(vert.attrib['y'])))
    return PolyGate(verts, axis, rect.attrib['name'])

                    
        
    return None
    
if __name__ == "__main__":
    import fcm
    a = load_flowjo_xml('/home/jolly/Projects/fcm/scratch/flowjoxml/pretty.xml')
    print a.file_names
    print a.gates
    print a.comp.keys(), '\n', a.comp['Comp Matrix']
    print a.tubes['Specimen_001_A1_A01.fcs'].comp[1] - a.comp['Comp Matrix'][1]
    print a.tubes['Specimen_001_A1_A01.fcs'].pprint()
    x = fcm.loadFCS('/home/jolly/Projects/fcm/scratch/flowjoxml/001_05Aug11.A01.fcs')
    print x.channels
    a.tubes['Specimen_001_A1_A01.fcs'].apply_gates(x)
    print x.tree.pprint()
    x.visit('foo1')
    print x.current_node.name, x.shape