import gclib
g=gclib.py()
c=g.GCommand

g.GOpen('192.168.255.200 --direct -s ALL')
print(g.GInfo())
