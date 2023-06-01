import numpy as np 
from param_matlab import param,m,NI
import gclib
import nidaqmx
from nidaqmx import stream_readers
from train import RL_loop
import matlab.engine
import time

g=gclib.py()
c=g.GCommand

#-----------------Connect to galil and set parameters ----------------------

g.GOpen('192.168.255.200 --direct -s ALL')
print(g.GInfo())

#Increase the acceleration and deceleration of the motor
ACF = 1.5
SPF = 1000000
KSF = 0.5 #Between 0.25 (low smoothing) and 64 (high smoothing)

c(f"ACF={round(256000 * ACF)}") 
c(f"DCF={round(256000 * ACF)}") 
c(f"SPF={SPF}") #Default value: 10666
c(f"KSF={KSF}") #Default value: 10666
c("DEF=0") #Force F encoder to 0 (because it's homed) to put after homing

#code from set_galil_analog_inputs.m

""" 
 * Set the analog input parameters properly
 Some channels are set to work in differential, some in single-ended
 outputs. In this case, the channels 3, 4 and 5 are set in differential,
 because they are more critical physically (3 and 4 are important for the
 power coefficient), and channels 1 and 2 in single-ended:
   - Single-ended inputs: channels 1 and 2
   - Differential inputs: channels 3, 4 and 5.
 Also, the input range is reduced to +- 5V to increase the accuracy."""

#Channel 1 and 2, read directly
#Channel 1
c("AQ1, 1")
#Channel 2
c("AQ2, 1")
#Channels 3, 4, 5: differential inputs
# Channel 3

c("AQ3, -1")
c("AQ4, 1")
# Channel 4
c("AQ5, -1")
c("AQ6, 1")
# Channel 5
c("AQ7, -1")
c("AQ8, 1")


#--------------------------------

RL_loop(c)





g.GClose()
