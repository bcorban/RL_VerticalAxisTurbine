# -*- coding: utf-8 -*-

import scipy.io as spio
import numpy as np
import matlab.engine
import getpass

eng = matlab.engine.start_matlab()

user=getpass.getuser()

if user=='PIVUSER':
    path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"
    eng.addpath (path, nargout= 0 )
    
elif user == 'adminit':
    path = "/home/adminit/RL_VerticalAxisTurbine/Carousel"
    eng.addpath (path, nargout= 0 )
    path = "/home/adminit/MATLAB_scripts"
    eng.addpath (path, nargout= 0 )


def load_mat(filename, var=None):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    if var:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True, variable_names=var)
    else:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    return _check_keys(data)

def get_param():
    eng.param_definition(nargout=0)
    # param=load_mat("/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/param.mat")['param']
    param=load_mat("./param.mat")['param']
    eng.quit()
    return param
    
        

if __name__=="__main__":

    print(get_param())
