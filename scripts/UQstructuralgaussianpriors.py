#####################################################
# Main script for applying SGP
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2022
# =================================================================
#####################################################

import numpy as np
import random
import os
import time
import phantomlib
import geometrylib
import gaussianpriorlib
import scipy as sp
import scipy.io as spio
import UQplots2 as UQplots
import UQpostprocess as UQpp
from UQcgls import CGLS_reg_samples, CGLS_reg_MAP, CGLS_reg_ML, CGLS_nolike_samples


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
rnl = 0.02                     # relative noise level
ag = "sparseangles20percent"                    # Problem geometry
if realdata == True:
    data_std = 0.05
    datapath = '../../FORCE/data/Data_20180911/'
else: 
    datapath = '../../FORCE/data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)

# UQ problem
likelihood = 'IIDGauss'              # type of likelihood 'IIDGauss', 'Nolike'
prior = 'StructuralGaussianGMRF'     # Prior distribution, choose 'Noprior', 'GMRF', 'StructuralGaussianGMRF',

# Expected linear attenuation coefficients
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
# Prior GMRF precision
prec_MRF = 2000

# Samplers
n_s = int(2e3)              # number of saved samples
n_b = int(0.2*n_s)          # burn-in
n_t = n_s+n_b               # total number of saved samples
n_u = 1                     # number of samples between saving
x_tol, n_cgls, ncgls_init = 1e-4, 10, 7    # for CGLS sampling
msgno = 1                   # No. of saved samples between printing progress

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
path1 = '../../../../../../work3/swech/' + ag + '/' 
path2 = 'rnl{:d}_precpipe{:d}_precout{:d}_precMRF{:d}/'.format(int(rnl*100), int(prec_vals[3]), int(prec_vals[4]), int(prec_MRF))
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
domain      = 55         # physical size of image domain
N           = 500        # reconstruction of NxN pixels

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
    datafile = 'sinoN8.dat'
    sino = np.loadtxt(datapath + datafile, delimiter=';')
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

# =================================================================
# Prior
# =================================================================

# setup weighting mask and return mask, mean, and precision for the SGP
mask, mu_prior, w, Wsq, D1, D2 = getattr(gaussianpriorlib, prior)(N, domain, maskradii, maskcenterinner, maskcenterouter,maskid, mu_vals, prec_vals, bndcond = 'zero')
R = sp.sparse.vstack([np.sqrt(prec_MRF)*D1, np.sqrt(prec_MRF)*D2, Wsq])
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

np.random.seed(1000)
print('\n***MCMC***\n')
f = open(path + "log.txt", "w")
f.write('\n***MCMC***\n')
f.close()

st = time.time()

x_s = np.zeros((N**2, n_t), dtype = 'f')
x_s[:, 0] = x0
for s in range(n_t-1):
    
    # draw a sample
    if likelihood == "Nolike":
        x_s[:, s+1], _ = CGLS_nolike_samples(x_s[:, s], R, Rmu_prior, b_data, lambd, n_cgls, x_tol)
    else:
        x_s[:, s+1], _ = CGLS_reg_samples(x_s[:, s], R, Rmu_prior, b_data, lambd, n_cgls, x_tol)

    # msg
    if (np.mod(s, msgno) == 0):
        print("\nSample {:d}/{:d}".format(s, n_t))
        f = open(path + "log.txt", "a")
        f.write('\nSample {:d}/{:d}'.format(s, n_t))
        f.close()

poteval_s = np.nan
acc_s = np.nan

print('\nElapsed time:', time.time()-st, '\n') 
f = open(path + "log.txt", "a")
f.write('\nElapsed time: {:f} \n'.format(time.time()-st)) 
f.close()

#%%=======================================================================
# Post processing
#=========================================================================

print('Post processing...')
f = open(path + "log.txt", "a")
f.write('Post processing...')
f.close()

# Chains for further analysis
chainno = np.array([5000, 43185])

# Integrated autocorrelation time
chainno_iact = random.sample(range(1, N**2+1), 100)
x_chains = x_s[chainno, :]      # pick out chains for visualization
tau_list, _ = UQpp.iact(x_s[chainno_iact, :].T)  # Autocorrelation time
tau, _ = UQpp.iact(x_chains.T)  # Autocorrelation time
tau_max = np.ceil(1)

# Burnthin
x_chains_thin = UQpp.burnthin(x_chains, n_b, tau_max)
x_thin = UQpp.burnthin(x_s, n_b, tau_max)

# Autocorrelation function
acf = np.zeros((n_s, len(chainno)))
for count, value in enumerate(chainno):
    acf[:, count] = UQpp.autocorr_func_1d(x_thin[value, :], norm=True) # Autocorrelation function

# Relative error
if realdata == False:
    x_e = UQpp.relative_error(x_s, x_truef, xt_norm, n_t)
    x_e_thin = UQpp.burnthin(x_e, n_b, tau_max)

# Posterior realizations
npost = 6
rseed = 0
post_realiz, post_idx = UQpp.posterior_realizations(x_thin, npost, rseed)

# Statistics
quant = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
x_mean, x_std, x_q = UQpp.statistics(x_thin, quant)

# Mean RSME
if realdata == False:
    RMSE_im = UQpp.RMSE(N, x_mean, x_truef)
    RMSE_pipe = UQpp.RMSE(np.count_nonzero(mask!=5), x_mean[mask!=5], x_truef[mask!=5])
    # msg
    print("\nRMSEim: {}\nRMSEpipe: {}".format(RMSE_im, RMSE_pipe))
    f = open(path + "log.txt", "a")
    f.write("\nRMSEim: {}\nRMSEpipe: {}".format(RMSE_im, RMSE_pipe))
    f.close()

# likelihood of mean and phantom
sino_xmean = UQpp.posterior_sinograms(x_mean)
loglike_xmean = UQpp.Gaussian_loglike(sino_xmean, b_data, lambd)
print('Log-likelihood of posterior mean is {:f}.'.format(loglike_xmean))
if realdata == False:
    loglike_xtrue = UQpp.Gaussian_loglike(b_true, b_data, lambd)
    print('Log-likelihood of true x is {:f}.'.format(loglike_xtrue))
else:
    loglike_xtrue = None


#%%=======================================================================
# Saving
#=========================================================================

mdict={'prior': prior,
        'x_mean': x_mean, 
        'x_std': x_std,
        'x_q': x_q, 
        'quant': quant,
        'post_realiz': post_realiz,
        'post_idx': post_idx,
        'npost': npost,
        'x_e': [], 
        'x_e_thin': [],
        'RMSE_im': [],
        'RMSE_pipe': [],
        'lambd': lambd,
        'x_chains': x_chains,
        'x_chains_thin': x_chains_thin,
        'chainno': chainno,
        'N': N,
        'n': [],
        'n_s': n_s,
        'n_b': n_b, 
        'n_u': n_u,
        'acc': acc_s,
        'tau': tau,
        'tau_list': tau_list,
        'chainno_iact': chainno_iact,
        'acf': acf,
        'tau_max': tau_max,
        'phantom': [], 
        'radii': [],
        'x_true': [],
        'x0': x0,
        'b_data': b_data,
        'b_true': [],
        'noise': [],
        'p': p,
        'q': q,
        'rnl': rnl,
        'mu_vals': mu_vals,
        'prec_vals': prec_vals,
        'prec_MRF': prec_MRF,
        'maskradii': maskradii,
        'maskcenterinner': maskcenterinner,
        'maskcenterouter': maskcenterouter,
        'mask': mask, 
        'loglike_xmean': loglike_xmean,
        'loglike_xtrue': [],
        'file': file,
        'path': path, 
        'realdata': realdata}

if realdata == False:
    mdict["phantom"].append(phantom)
    mdict["n"].append(n)
    mdict["x_true"].append(x_true)
    mdict["radii"].append(radii)
    mdict["b_true"].append(b_true)
    mdict["noise"].append(noise)
    mdict["loglike_xtrue"].append(loglike_xtrue)
    mdict["RMSE_im"].append(RMSE_im)
    mdict["RMSE_pipe"].append(RMSE_pipe)
    mdict["x_e"].append(x_e)
    mdict["x_e_thin"].append(x_e_thin)

print('Saving...')
f = open(path + "log.txt", "a")
f.write('Saving...')
f.close()
spio.savemat(path + file + '.mat', mdict)

#%%=======================================================================
# Saving
#=========================================================================

print('Plotting...')
f = open(path + "log.txt", "a")
f.write('Plotting..')
f.close()


slice_vertical = 0
slice_horizontal = -18.5

cmin_im = -0.05
cmax_im = 0.2

cmin_sino = None
cmin_sino = None

UQplots.iact(chainno_iact, tau_list, 'iact', path)
UQplots.acf(acf, tau, chainno, path)
UQplots.image2D(x0, N, domain, 'Initial Guess', path, 'x0', cmin=cmin_im, cmax=cmax_im)
UQplots.image2D(x_mean, N, domain, 'Posterior Mean', path, 'posterior_mean', cmin=cmin_im, cmax=cmax_im)
UQplots.image2D(x_std, N, domain, 'Posterior std', path, 'posterior_std', cmin=None, cmax=None)
UQplots.image2D(x_q[:,4]-x_q[:,0], N, domain, '95 % CI width', path, 'interq95_range', cmin=None, cmax=None)
for i in range(5):
    UQplots.image2D(x_q[:,i], N, domain, '{} % quantile'.format(quant[i]*100), path, 'xq_{}'.format(quant[i]*100), cmin=cmin_im, cmax=cmax_im)
for i in range(6):
    UQplots.image2D(post_realiz[:,i], N, domain, 'Posterior realization no. {}'.format(post_idx[i]), path, 'posterior_realiz{}'.format(post_idx[i]), cmin=cmin_im, cmax=cmax_im)

if realdata == True:
    x_truef = None
UQplots.slices1Dvs3(x_mean, x_q[:,0], x_q[:,4], x_truef, N, domain, slice_vertical, slice_horizontal, realdata, path, cmin=cmin_im, cmax=cmax_im)
UQplots.slices1D_postreals(post_realiz, post_idx, x_mean, x_truef, N, domain, slice_vertical, slice_horizontal, realdata, path, cmin=cmin_im, cmax=cmax_im)

UQplots.xchains(x_chains, x_chains_thin, chainno, tau_max, 'x_chains', path)
if realdata == False:
    UQplots.image2D(phantom.flatten(order='F'), int(np.sqrt(len(phantom.flatten()))), domain, 'Phantom', path, 'phantom', cmin=cmin_im, cmax=cmax_im)
    UQplots.image2D(x_mean-x_truef, N, domain, 'Error', path, 'imerror', cmin=None, cmax=None)
    UQplots.error_chain(x_e, x_e_thin, path)

UQplots.sino2(b_data, p, q, path, 'sino_noisy', cmin = None, cmax = None)

UQplots.imagemovie(x_s[:,4::], domain, N, n_s, 1, path, "samples_movie", colmap = 'gray', cmin = cmin_im, cmax = cmax_im)
    

print('Done')
f = open(path + "log.txt", "a")
f.write('Done')
f.close()
