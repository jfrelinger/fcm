"""Example of chaining methods on object."""

class Foo(object):
    def __init__(self):
        self.items = []
        self.num = 0

    def f1(self, x):
        return f1(self, x)

    def f2(self, x):
        return f2(self, x)

    def f3(self, x):
        return f3(self, x)

def f1(foo, x):
    print "f1"
    foo.items.append('f1')
    foo.num += x
    return foo

def f2(foo, x):
    print "f2"
    foo.items.append('f2')
    foo.num += x
    return foo

def f3(foo, x):
    print "f3"
    foo.items.append('f3')
    foo.num += x
    return foo

if __name__ == '__main__':
    foo = Foo()
    foo = foo.f1(3).f2(7).f3(2)
    print foo.num
    print foo.items
