import networkx

class Tree(object):
    def __init__(self):
        self.g = networkx.LabeledDiGraph()
        self.g.add_node('root')
        self.current = 'root'

    def parent(self):
        return self.g.predecessors(self.current)[0]

    def children(self):
        return self.g.successors(self.current)

    def visit(self, name):
        self.current = name

    def get(self):
        return self.g.get_node(self.current)

    def add_child(self, name, data):
        self.g.add_node(name, data)
        self.g.add_edge(self.current, name)
        self.current = name

if __name__ == '__main__':
    t = Tree()
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
    
    
