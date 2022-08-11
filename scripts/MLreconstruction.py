#####################################################
# Reconstruction for initializing SGP algorithm
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2022
# =================================================================
#####################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import phantomlib
import geometrylib
import scipy.io as spio
import UQplots2 as UQplots
import UQpostprocess as UQpp
from UQcgls import CGLS_reg_ML


import types
import functools
def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

#%%=======================================================================
# Parameters
#=========================================================================
# Data 
realdata = True
phantomname = 'DeepSeaOilPipe4'   # choose phantom
rnl = 0.02                      # relative noise level
ag = "sparseangles20percent"                    # Problem geometry
if realdata == True:
    data_std = 0.05
    datapath = '../data/Data_20180911/sinoN8.dat'
else:
    datapath = '../data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)

# CGLS params
x_tol, n_cgls_vec = 1e-6, np.linspace(1,20,20)   # for CGLS sampling

# Filepaths
path1 = '../output/' + ag + '/' 
path2 = 'rnl{:d}_CGLS_MLsweep/'.format(int(rnl*100))
path = path1 + path2
file = 'UQout'
os.makedirs(path, exist_ok=True)

#%%=======================================================================
# Define geometry
#=========================================================================
# Aqusition geometry
p, theta, stc, ctd, shift, vectors, dl, dlA = geometrylib.Data20180911(ag)
q = len(theta)

# Reconstruction geometry
domain      = 55          # physical size of object
N           = 512         # reconstruction of NxN pixels

#%%=======================================================================
# Create/load sinogram
#=========================================================================
if realdata == False:
    data = spio.loadmat(datapath + 'data.mat')
    lambd = data['lambd'][0][0]
    n = data['n'][0][0]
    phantom = data['phantom']
    sino_astra = data['sino_astra']
    b_true = data['b_true'][0]
    b_data = data['b_data'][0]
    noise = data['noise'][0]

    # underlying true in same dimension as reconstruction
    x_true, radii = getattr(phantomlib, phantomname)(N,True)
    x_truef = x_true.flatten(order='F')
    xt_norm = np.linalg.norm(x_truef)

else:
    # load data and change to correct data structure
    sino = np.loadtxt(datapath, delimiter=';')
    sino = sino.astype('float32')
    sino_astra = np.rot90(sino, k = 1)
    sino_astra = sino_astra[:-8, :]
    print(np.min(sino))
    if ag == 'sparseangles50percent':
        sino_astra  = sino_astra[::4, :]
    elif ag == 'sparseangles20percent':
        sino_astra  = sino_astra[::10, :]
    elif ag == 'sparseangles':
        sino_astra  = sino_astra[::20, :]
    elif ag == 'full':
        sino_astra  = sino_astra[::2, :]
    b_data = sino_astra.flatten()
    m = len(b_data)

    # compute noise parameters
    noise_std = data_std
    lambd = 1/noise_std**2

    # variables not defined for true data
    phantom = np.nan
    n = np.nan
    x_true = np.array([])
    x_truef = np.nan
    xt_norm = np.nan
    radii = np.nan
    b_true = np.nan
    noise = np.nan


#%%=======================================================================
# ML reconstruction
#=========================================================================

# ML
print('\n***ML***\n')
f = open(path + "log.txt", "w")
f.write('\n***ML***\n')
f.close()

st = time.time()

x_ML = np.zeros((N**2, len(n_cgls_vec)), dtype = 'f')
RMSE_im = np.zeros(len(n_cgls_vec), dtype = 'f')
cmin_im = -0.05
cmax_im = 0.2

x_ML_prev = np.zeros(N**2)
    
# Sweep
for idx, n_cgls in enumerate(n_cgls_vec):

    # Compute MAP   
    x_ML[:, idx], _ = CGLS_reg_ML(np.zeros(N**2), b_data, lambd, n_cgls, x_tol)
    UQplots.image2D(x_ML[:, idx], N, domain, 'MAP', path, 'ML_{}iter'.format(int(n_cgls)), cmin=cmin_im, cmax=cmax_im)
    # Mean RSME
    if realdata == False:
        RMSE_im[idx] = UQpp.RMSE(N, x_ML[:, idx], x_truef)

    x_ML_prev = x_ML[:, idx]
    # msg
    print("\nCGLS iter: {:d}\nRMSEim: {}".format(int(n_cgls), RMSE_im[idx]))
    f = open(path + "log.txt", "a")
    f.write("\nCGLS iter: {:d}\nRMSEim: {}".format(int(n_cgls),  RMSE_im[idx]))
    f.close()

print('\nElapsed time:', time.time()-st, '\n') 
f = open(path + "log.txt", "a")
f.write('\nElapsed time: {:f} \n'.format(time.time()-st)) 
f.close()

if realdata == False:
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(3)
    plt.plot(n_cgls_vec,RMSE_im, lw = 2)
    plt.xlabel('CGLS iterations')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'RMSEim.png')


mdict={'x_ML': x_ML,
        'RMSE_im': RMSE_im,
        'N': N,
        'phantom': phantom, 
        'radii': radii,
        'x_true': x_true,
        'b_data': b_data,
        'noise': noise,
        'p': p,
        'q': q,
        'rnl': rnl,
        'file': file,
        'path': path, 
        'realdata': realdata}

print('Saving...')
f = open(path + "log.txt", "a")
f.write('Saving...')
f.close()
spio.savemat(path + file + '.mat', mdict)

print('Done!')
f = open(path + "log.txt", "a")
f.write('Done!')
f.close()

