'''
Created on Feb 11, 2012

@author: Jacob Frelinger
'''

import xml.etree.cElementTree as xml
import numpy
import bisect
from fcm import PolyGate

class InterpTable(object):
    '''
    a look up table that linearly interperlates between values
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, v):
        j = bisect.bisect_left(self.x, v)
        i = j-1
        if i < 0 :
            return self.y[0]
        if j >= len(self.x) :
            return self.y[ -1 ]
        return self.y[i] + (v-self.x[i])*(self.y[j]-self.y[i])/(self.x[j]-self.x[i])
        

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
        #print self.gate.name, self.gate.chan
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
    
    table = root.find('CalibrationTables')
    if table:
        table = parse_table(table[0])
    
        
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
                        else:
                            prefix, suffix = (None, None)
        sample = node.find('SampleNode')
        if comp_matrix: 
            fcsfile = xmlfcsfile(sample.attrib['name'], comp = comp_matrix)
        else:
            fcsfile = xmlfcsfile(sample.attrib['name'])                      
        # find gates
        fcsfile.pops = find_pops(sample, prefix, suffix, table)
        files[fcsfile.name] = fcsfile
                            
    if len(comps) > 0:
        return FlowjoWorkspace(files,comps)
    else:
        return FlowjoWorkspace(files)
 
 
def find_pops(node, prefix=None, suffix=None, table=None):
    pops = {}
    for i in node:
        if i.tag == 'Population':
            pops[i.attrib['name']] = build_pops(i, prefix, suffix, table)
    return pops
    
    
def build_pops(node,prefix=None, suffix=None, table=None):
    #print "*"*depth, node.tag, node.attrib['name']
    name = node.attrib['name']
    
    children = {}
    for i in node:
        if i.tag == 'Population':
            tmp = build_pops(i, prefix, suffix, table)
            if tmp is not None:
                children[i.attrib['name']] = tmp
            
        elif i.tag == 'PolygonGate':
            for j in i:
                if j.tag == 'PolyRect':
                    g = build_Polygon(j, prefix, suffix, table)
                    #print g.chan
                elif j.tag == 'Polygon':
                    g = build_Polygon(j, prefix, suffix, table)
                    #print g.chan
    try:            
        return  PopulationNode(name, g, children)
    except UnboundLocalError:
        return None
    

def build_Polygon(rect, prefix=None, suffix=None, table=None):
    verts = []
    axis = [rect.attrib['xAxisName'], rect.attrib['yAxisName']]
    #print axis, prefix, suffix, 
    replaced = [False, False]
    if prefix is not None:
        for i,j in enumerate(axis):
            if j.startswith(prefix):
                axis[i] = j.replace(prefix,'')
                #replaced[i] = True
    if suffix is not None:
        for i,j in enumerate(axis):
            if j.endswith(suffix):
                axis[i] = j[:-(len(suffix))]
                #replaced[i] = True
    axis = tuple(axis)
    #print axis
    for vert in rect.iter('Vertex'):
        if replaced[0]:
            print 'reversing x'
            x = table[(float(vert.attrib['x']))]
        else:
            x = float(vert.attrib['x'])
        if replaced[1]:
            print 'reversing y'
            y = table[(float(vert.attrib['y']))]
        else:
            y = float(vert.attrib['y'])
        verts.append((x,y))
    print rect.attrib['name'], axis, verts
    return PolyGate(verts, axis, rect.attrib['name'])

                    
def parse_table(table):
    tmp = table.text.split(',')
    count = len(tmp)/2
    xs = []
    ys = []
    for i in range(count):
        ys.append(float(tmp[i*2]))
        xs.append(float(tmp[i*2+1]))
    transform = InterpTable(xs,ys)
    return transform
    
if __name__ == "__main__":
    import fcm
    a = load_flowjo_xml('/home/jolly/Projects/fcm/scratch/flowjoxml/pretty.xml')
    print a.file_names
    print a.gates
    print a.comp.keys(), '\n', a.comp['Comp Matrix']
    sidx, spill = a.comp['Comp Matrix']
    print a.tubes['Specimen_001_A1_A01.fcs'].comp[1] - a.comp['Comp Matrix'][1]
    print a.tubes['Specimen_001_A1_A01.fcs'].pprint()
    x = fcm.loadFCS('/home/jolly/Projects/fcm/scratch/flowjoxml/001_05Aug11.A01.fcs', auto_comp=False, transform=None)#, sidx=sidx, spill=spill, transform='logicle')
    x.logicle()
    x.compensate(sidx=sidx,spill=spill)
    print x.channels
    a.tubes['Specimen_001_A1_A01.fcs'].apply_gates(x)
    print x.tree.pprint(size=True)
    x.visit('foo1')
    print x.current_node.name, x.shape, 238712
    x.visit('cd4')
    print x.current_node.name, x.shape, 43880