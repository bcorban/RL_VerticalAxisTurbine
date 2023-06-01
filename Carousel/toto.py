import gclib
import numpy as np
# g=gclib.py()
# c=g.GCommand

# g.GOpen('192.168.255.200 --direct -s ALL')
# print(g.GInfo())

def filloutliers(ws,volts,volts_raw,offset):
    window=volts[-ws:]
    med=np.median(window)
    MAD=1.4826*np.median(np.abs(window-med))
    if np.abs(volts[-1]-med)>3/MAD:
        volts[-1]=med+3/MAD*np.sign(volts[-1]-med)
    if window==np.ones(ws)*med:
        volts[-ws:]=-(volts_raw[-ws]-offset)
