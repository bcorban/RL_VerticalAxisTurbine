import gclib
import matlab.engine
import numpy as np
from param_matlab import param,m,NI

def home(eng):
    print('homing...')
    eng.my_quick_home(nargout=0)
    

def pitch(action_abs,g):

    print('pitching')
    g.GCommand(f"PAF={action_abs}")


def read_state(g,offset):
    c=g.GCommand
    print('state')
    galil_output=c('MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TDE,_TDF,_TPF')
    volts_raw=galil_output[0:5]
    volts=-(volts_raw)-offset
    phase= galil_output['phase_index'] / m[1]['ms'] % 360# In degrees
    phase_cont = galil_output(6) / m(1).ms #In degrees
        
    
    
    
    
    
    
    
    
    
    
    
    return np.zeros(3)

def start_E(g):
    print("Starting motor E")
    g.GCommand('SHE')
    g.GCommand(f"JGE={param['JG']}")
    g.GCommand('BGE')

    #initialize position tracking
    g.GCommand("PTF=1")

def stop_E(g):
    print("Stopping motor E")
    g.GCommand("ST")

def wait_N_rot(g,N):
    print(f"waiting for {N} periods")
    phase_ini=read_state(g)['phase_index']//360
    phase=phase_ini
    while phase-phase_ini< N:
        phase=read_state(g)['phase_index']//360
        
def save_data():
    print("saving data")