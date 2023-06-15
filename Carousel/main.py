# from param_matlab import param, m, NI
import gclib
# from train import RL_loop_sb3,RL_loop_rllib
from setup_galil import setup_g
import getpass
from sac_continuous_action import clean_RLloop
g = gclib.py()

# -----------------Connect to galil and set parameters ----------------------


user=getpass.getuser()
if user=='PIVUSER':
  g.GOpen("192.168.255.200 --direct -s ALL")

elif user == 'adminit':
  g.GOpen("192.168.255.25 --direct -s ALL")
  

setup_g(g)

# --------Start RL training loop-------------------

# RL_loop_sb3()
clean_RLloop()

# --------------------------------

g.GClose()
