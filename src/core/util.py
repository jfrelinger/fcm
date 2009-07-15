import networkx

class Tree(object):
    def __init__(self, pnts):
        self.g = networkx.LabeledDiGraph()
        self.root = RootNode('root', pnts)
        self.g.add_node('root', self.root)
        self.current = 'root'

    def parent(self):
        return self.g.predecessors(self.current)[0]

    def children(self):
        return self.g.successors(self.current)

    def visit(self, name):
        self.current = name

    def get(self):
        return self.g.get_node(self.current)
    
    def view(self):
        return self.g.get_node(self.current).view()

    def add_child(self, name, node):
        self.g.add_node(name, node)
        self.g.add_edge(self.current, name)
        self.current = name


class Node(object):
    """
    base node object
    """
    
    def __init__(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
    
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
        
class TransformNode(Node):
    """
    Transformed Data Node
    """
    
    def __init__(self, name, parent,  data):
        self.name = name
        self.parent = parent
        self.data = data
        
class GatingNode(Node):
    """
    Node of gated data
    """
    
    def __init___(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
        
    def view(self):
        return self.parent.view()[self.data]
        
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = Tree([0,0,0])
    t.add_child('gate1', [1,2,3])
    t.add_child('gate11', [4,5,6])
    t.visit('root')
    t.add_child('gate2', [2,3,4])
    t.add_child('gate21', [3,4,5])
    t.add_child('gate211', [4,5,6])

    parent = t.parent()
    print t.current
    print t.parent()
    t.visit(t.parent())
    print t.current
    print t.parent()
    t.visit('root')
    print t.current
    print t.children()
    print t.get()
    t.visit('gate2')
    print t.get()
    networkx.draw(t.g)
    plt.show()
    
    
