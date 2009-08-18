import networkx
import re

class Tree(object):
    '''Tree of data for FCMdata object.'''
    def __init__(self, pnts):
        self.g = networkx.LabeledDiGraph()
        self.root = RootNode('root', pnts)
        self.g.add_node('root', self.root)
        self.current = 'root'

    def parent(self):
        '''return the parent of a node'''
        return self.g.predecessors(self.current)[0]

    def children(self):
        '''return the children of a node'''
        return self.g.successors(self.current)

    def visit(self, name):
        '''visit a node in the tree'''
        self.current = name

    def get(self):
        '''return the current node object'''
        return self.g.get_node(self.current)
    
    def view(self):
        '''Return a view of the current data'''
        return self.g.get_node(self.current).view()

    def add_child(self, name, node):
        '''Add a node to the tree at the currently selected node'''
        if name == '':
            prefix = node.prefix
            pat = re.compile(prefix + "(\d+)")
            matches = [pat.search(i) for i in self.g.nodes()]
            matches = [i for i in matches if i is not None]
            n = max([ int(i.group(1)) for i in matches])
            name = prefix + str(n+1)
        self.g.add_node(name, node)
        self.g.add_edge(self.current, name)
        self.current = name
        
    def rename_node(self, old_name, new_name):
        '''Rename a node from old_name to new_name'''
        node = self.g.get_node(old_name)
        pred = self.g.predecessors(old_name)
        children = self.g.successors(old_name)
        self.g.remove_node(old_name)
        node.name = new_name
        self.g.add_node(new_name, node)
        for i in pred:
            self.g.add_edge(i, new_name)
        for i in children:
            self.g.add_edge(new_name, i)
            
        
class Node(object):
    """
    base node object
    """
    
    def __init__(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
        self.prefix= 'n'
    
    def view(self):
        """
        return the view of the data associated with this node
        """
        
        return self.data
    
class RootNode(Node):
    """
    Root Node
    """
    
    def __init__(self, name, data):
        self.name = name
        self.parent = None
        self.data = data
        self.prefix='root'
        
class TransformNode(Node):
    """
    Transformed Data Node
    """
    
    def __init__(self, name, parent,  data):
        self.name = name
        self.parent = parent
        self.data = data
        self.prefix = 't'
        
class SubsampleNode(Node):
    """
    Node of subsampled data
    """
    
    def __init__(self, name, parent, param):
        self.name = name
        self.parent = parent
        self.param = param
        self.prefix = 's'
        
    def view(self):
        return self.parent.view().__getitem__(self.param)
    
class GatingNode(Node):
    """
    Node of gated data
    """
    
    def __init___(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
        self.prefix = 'g'
        
    def view(self):
        return self.parent.view()[self.data]
        
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = Tree(RootNode('root',[0,0,0]))
    t.add_child('gate1', RootNode('gate1',[1,2,3]))
    t.add_child('gate11', RootNode('gate11',[4,5,6]))
    t.visit('root')
    t.add_child('gate2', RootNode('gate2',[2,3,4]))
    t.add_child('gate21', RootNode('gate21',[3,4,5]))
    t.add_child('gate211', RootNode('gate211',[4,5,6]))

    parent = t.parent()
    print t.current
    print t.parent()
    t.visit(t.parent())
    print t.current
    print t.parent()
    t.visit('root')
    print t.current
    print t.children()
    print t.view()
    t.visit('gate2')
    print t.view()
    networkx.draw(t.g)
    plt.show()
    
    t.rename_node('gate1', 'foo')
    t.visit('foo')
    print t.current
    print t.view()
    networkx.draw(t.g)
    plt.show()
    
    
    
