#####################################################
# Finding good GMRF precision values for SGP prior
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
import gaussianpriorlib
import scipy as sp
import scipy.io as spio
import UQplots2 as UQplots
import UQpostprocess as UQpp
from UQcgls import CGLS_reg_samples, CGLS_reg_MAP, CGLS_reg_ML


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
realdata = False
phantomname = 'DeepSeaOilPipe4'   # choose phantom
data_std = 0.05
rnl = 0.02                      # relative noise level
ag = "sparseangles20percent"                    # Problem geometry
if realdata == True:
    data_std = 0.05
    datapath = '../../FORCE/data/Data_20180911/'
else: 
    datapath = '../../FORCE/data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)

# UQ problem
likelihood = 'IIDGauss'          # type of lieklihood 'IIDGauss', 'Nolike'
prior = 'StructuralGaussianGMRF'     # Prior distribution, choose 'Noprior', 'GMRF', 'StructuralGaussianGMRF',

# CGLS params
x_tol, n_cgls, ncgls_init = 1e-4, 100, 7    # for CGLS sampling

steel = 2e-2*7.9
air = 0
PUrubber = 5.1e-2*0.94
PEfoam = 5.1e-2*0.15
concrete = 4.56e-2*2.3
# Prior means
mu_vals = np.array([steel, PEfoam, PUrubber, concrete, air])
# Prior precision values for IID part of prior. Comment lines to choose SGP configuration.
prec_vals = np.array([1000, 1000, 1000, 500, 1000]) # SGP-F
#prec_vals = np.array([0, 0, 0, 0, 1000]) # SGP-BG
#prec_vals = np.array([0, 0, 0, 0, 0]) # GMRF
# Prior GMRF precision sweep
prec_MRF_vec = np.logspace(2, 5, 20, endpoint=True) # vector

if realdata == False:
    # Synthetic
    maskid = np.array([5,1,2,3,4,5])
    piperadii = np.array([9,11,16,17.5,23])
    maskradii = np.array([[0,piperadii[0]-0.5],
                            [piperadii[0]+0.5, piperadii[1]-0.5],
                            [piperadii[1]+0.5, piperadii[2]-0.5],
                            [piperadii[2]+0.5, piperadii[3]-0.5],
                            [piperadii[3]+0.5, piperadii[4]-0.5],
                            [piperadii[4]+0.5, 50]])
    maskcenterinner = np.zeros((6,2))
    maskcenterouter = np.zeros((6,2))
else:
    # Real data
    maskid = np.array([5,1,2,3,4,5,0,0,0,0,0])
    maskradii = np.array([[0,8.7],
                            [9.7,10.3],
                            [11.3,15],
                            [16.5,17.1],
                            [18.1,22],
                            [23,40],
                            [0,1.5],
                            [0,1.5],
                            [0,1.5],
                            [0,1.5],
                            [0,1.5]])
    maskcenterinner = np.array([[0,0],
                            [0.6,0.8],
                            [0.6,0.8],
                            [0,0.4],
                            [0,0.4],
                            [0,0.4],
                            [-7,4],
                            [23.4,3.7],
                            [-23.1,2.5],
                            [0.8,-23],
                            [1,24.1]])
    maskcenterouter= np.array([[0.6,0.8],
                            [0.6,0.8],
                            [0,0.4],
                            [0,0.4],
                            [0,0.4],
                            [0,0.4],
                            [-7,4],
                            [23.4,3.7],
                            [-23.1,2.5],
                            [0.8,-23],
                            [1,24.1]])


# Filepaths
path1 = '../output/' + ag + '/' 
path2 = 'rnl{:d}_precpipe{:d}_precout{:d}_precMRFsweep/'.format(int(rnl*100), int(prec_vals[3]), int(prec_vals[4]))
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
domain      = 55        # physical size of object
N           = 500       # reconstruction of NxN pixels

#%%=======================================================================
# Create/load sinogram
#=========================================================================
if realdata == False:
    data = spio.loadmat(datapath + 'data.mat')
    lambd = data['lambd'][0][0]
    print(1/np.sqrt(lambd))
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
    sino = np.loadtxt(datapath + 'sinoN8.dat', delimiter=';')
    sino = sino.astype('float32')
    sino_astra = np.rot90(sino, k = 1)
    sino_astra = sino_astra[:-8, :]
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

# =================================================================
# Prior
# =================================================================

# setup weighting mask and return mask, mean, and precision for the weighted Gaussian prior
mask, mu_prior, w, Wsq, D1, D2 = getattr(gaussianpriorlib, prior)(N, domain, maskradii, maskcenterinner, maskcenterouter, maskid, mu_vals, prec_vals, bndcond = 'zero')
Rmu_prior = np.hstack([np.zeros(np.shape(D1)[0]), np.zeros(np.shape(D2)[0]), Wsq*mu_prior])

#%%=======================================================================
# Sample posterior
#=========================================================================

# init sampling
if likelihood == "Nolike":
    x0 = np.zeros(N**2)
#elif realdata==True:
x0, _ = CGLS_reg_ML(np.zeros(N**2), b_data, lambd, ncgls_init, x_tol)
# else:
#     MLrecon = spio.loadmat(datapath+'ML.mat')
#     x0 = MLrecon['x_ML'][:, ncgls_init-1]

# MAP
print('\n***MAP***\n')
f = open(path + "log.txt", "w")
f.write('\n***MAP***\n')
f.close()

st = time.time()

x_MAP = np.zeros((N**2, len(prec_MRF_vec)), dtype = 'f')
RMSE_im = np.zeros(len(prec_MRF_vec), dtype = 'f')
RMSE_pipe = np.zeros(len(prec_MRF_vec), dtype = 'f')
it = np.zeros(len(prec_MRF_vec), dtype = 'f')
cmin_im = -0.05
cmax_im = 0.2

# Sweep
for idx, prec_MRF in enumerate(prec_MRF_vec):

    # Update sqrtprec
    R = sp.sparse.vstack([np.sqrt(prec_MRF)*D1, np.sqrt(prec_MRF)*D2, Wsq])
    
    # Compute MAP   
    x_MAP[:, idx], it[idx] = CGLS_reg_MAP(x0, R, Rmu_prior, b_data, lambd, n_cgls, x_tol)
    UQplots.image2D(x_MAP[:, idx], N, domain, 'MAP', path, 'MAP_MRFprec{}'.format(prec_MRF), cmin=cmin_im, cmax=cmax_im)
    # Mean RSME
    if realdata == False:
        RMSE_im[idx] = UQpp.RMSE(N, x_MAP[:, idx], x_truef)
        RMSE_pipe[idx] = UQpp.RMSE(np.count_nonzero(mask!=5), x_MAP[mask!=5, idx], x_truef[mask!=5])

    # msg
    print("\nMAP {:d}/{:d}\nCGLS iter: {:d}\nGMRFprec: {}\nRMSEim: {}".format(idx+1, len(prec_MRF_vec), int(it[idx]), prec_MRF, RMSE_im[idx]))
    f = open(path + "log.txt", "a")
    f.write("\nMAP {:d}/{:d}\nCGLS iter: {:d}\nGMRFprec: {}\nRMSEim: {}".format(idx+1, len(prec_MRF_vec), int(it[idx]), prec_MRF, RMSE_im[idx]))
    f.close()

print('\nElapsed time:', time.time()-st, '\n') 
f = open(path + "log.txt", "a")
f.write('\nElapsed time: {:f} \n'.format(time.time()-st)) 
f.close()


fig, ax = plt.subplots(num=1, ncols=1, nrows=1, figsize=[16, 14], gridspec_kw=dict(hspace=0.2, wspace = 0.6), clear = True)
im = ax.imshow(mask.reshape(N,N), 
                cmap='Set2', 
                vmin = 0, 
                vmax = 5, 
                aspect='equal',
                interpolation = 'none')
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
fig.colorbar(im, cax=cax) 
plt.show()
plt.savefig(path + 'mask.png')
plt.savefig(path + 'mask.eps', format = 'eps')

if realdata == False:
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(3)
    plt.semilogx(prec_MRF_vec,RMSE_im, lw = 2)
    plt.xlabel('GMRF precision')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'RMSEim.png')
    plt.savefig(path + 'RMSEim.eps', format = 'eps')

    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(3)
    plt.semilogx(prec_MRF_vec,RMSE_pipe, lw = 2)
    plt.xlabel('GMRF precision')
    plt.ylabel('RMSE on the pipe')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'RMSEpipe.png')
    plt.savefig(path + 'RMSEpipe.eps', format = 'eps')


    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(3)
    plt.loglog(prec_MRF_vec,RMSE_im)
    plt.xlabel('GMRF precision')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'RMSEim_loglog.png')
    plt.savefig(path + 'RMSEim_loglog.eps', format = 'eps')


    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(3)
    plt.loglog(prec_MRF_vec,RMSE_pipe, lw = 2)
    plt.xlabel('GMRF precision')
    plt.ylabel('RMSE on the pipe')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'RMSEpipe_loglog.png')
    plt.savefig(path + 'RMSEpipe_loglog.eps', format = 'eps')


mdict={'x_MAP': x_MAP,
        'RMSE_pipe': RMSE_pipe,
        'RMSE_im': RMSE_im,
        'prior': prior,
        'N': N,
        'phantom': phantom, 
        'radii': radii,
        'x_true': x_true,
        'x0': x0,
        'b_data': b_data,
        'noise': noise,
        'p': p,
        'q': q,
        'rnl': rnl,
        'mu_vals': mu_vals,
        'prec_vals': prec_vals,
        'prec_MRF_vec': prec_MRF_vec,
        'maskradii': maskradii,
        'maskcenterinner': maskcenterinner,
        'maskcenterouter': maskcenterouter,
        'mask': mask, 
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

