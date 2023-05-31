import gclib
# g=gclib.py()
# c=g.GCommand

# g.GOpen('192.168.255.200 --direct -s ALL')
# print(g.GInfo())


class dummy():
    def __init__(self): 
        self.x=1
    def add(self):
        self.x+=1
    def test(self):
        for i in range(7):
            self.add()
        print(self.x)
        
a=dummy()
a.test()