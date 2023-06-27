
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import gclib
from setup_galil import setup_g
import getpass
from sac_continuous_action import clean_RLloop
g = gclib.py()
if __name__ == '__main__':
# -----------------Connect to galil and set parameters ----------------------


  user=getpass.getuser()
  if user=='PIVUSER':
    g.GOpen("192.168.255.200 --direct -s ALL")

  elif user == 'adminit':
    g.GOpen("192.168.255.25 --direct -s ALL")
    

  setup_g(g)  

# --------Start RL training loop-------------------

  clean_RLloop()

# --------------------------------

  g.GClose()
