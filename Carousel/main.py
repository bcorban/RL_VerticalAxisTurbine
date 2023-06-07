from param_matlab import param, m, NI
import gclib
from train import RL_loop
from setup_galil import setup_g

g = gclib.py()

# -----------------Connect to galil and set parameters ----------------------

g.GOpen("192.168.255.200 --direct -s ALL")
print(g.GInfo())
setup_g(g)

# --------------------------------

RL_loop()

# --------------------------------


g.GClose()
