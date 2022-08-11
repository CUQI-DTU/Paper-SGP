#####################################################
# Generate Synthetic data
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2022
# =================================================================
#####################################################

import numpy as np
import scipy.io as spio
import os
import phantomlib
import geometrylib
import astra
import UQplots2 as UQplots

#%%=======================================================================
# Parameters
#=========================================================================
# CT problem
phantomname = 'DeepSeaOilPipe4'   # choose phantom
rnl = 0.02                      # relative noise level
n = 1024                        # phantom dimension
ag = "sparseangles20percent"                    # Problem geometry

# reconstruction parameters
N = 512
niter = 400
domain_recon      = 55              # physical size of object

# Filepaths
path = '../data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)
os.makedirs(path, exist_ok=True)

#%%=======================================================================
# Define geometry
#=========================================================================
# Aqusition geometry
p, theta, stc, ctd, shift, vectors, dl, dlA = geometrylib.Data20180911(ag)
q = len(theta)
domain      = 55              # physical size of object

#%%=======================================================================
# Create/load sinogram
#=========================================================================
# create phantom
phantom, _ = getattr(phantomlib, phantomname)(n,True)

# geometries
vol_geom = astra.create_vol_geom(n,n,-domain/2,domain/2,-domain/2,domain/2)
proj_geom = astra.create_proj_geom('fanflat_vec', p, vectors)
proj_id0 = astra.create_projector('cuda', proj_geom, vol_geom) # line_fanflat

# create sinogram
_ , sino_astra = astra.create_sino(phantom, proj_id0)
# clean up
b_true = sino_astra.flatten()
m = len(b_true)  # dimension of data

# add noise
e0 = np.random.normal(0, 1, np.shape(b_true))
noise_std = rnl*np.linalg.norm(sino_astra)/np.linalg.norm(e0)
lambd = 1/noise_std**2
noise = noise_std*e0
b_data = b_true + noise

# underlying true in same dimension as reconstruction
x_true, radii = getattr(phantomlib, phantomname)(N,True)
x_truef = x_true.flatten(order='F')
xt_norm = np.linalg.norm(x_truef)


#%%=======================================================================
# Astra SART reconstruction
#=========================================================================

# Starting point
#geometries
vol_geom = astra.create_vol_geom(N,N,-domain_recon/2,domain_recon/2,-domain_recon/2,domain_recon/2)
proj_geom = astra.create_proj_geom('fanflat_vec', p, vectors)
# data objects
sino_id     = astra.data2d.create('-sino', proj_geom, sino_astra)   # sino object
rec_id      = astra.data2d.create('-vol', vol_geom)           # recon object
# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SART_CUDA') # SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sino_id
cfg['option']={}
#cfg['option']['MinConstraint'] = 0
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, niter)
# retrieve reconstruction and save it
rec = astra.data2d.get(rec_id)
x_recon = rec.flatten(order='F')
# clean up
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sino_id)

#%%=======================================================================
# Save data
#=========================================================================

mdict={'phantomname': phantomname,
        'ag': ag,
        'rnl': rnl,
        'lambd': lambd,
        'n': n,
        'phantom': phantom, 
        'b_data': b_data,
        'sino_astra': sino_astra,
        'b_true': b_true,
        'noise': noise,
        'p': p,
        'q': q,
        'path': path}

recon_dict = {'N': N,
                'x_recon': x_recon, 
                'x_true': x_true, 
                'x_truef': x_truef, 
                'radii': radii}

spio.savemat('{}recon_N{:d}.mat'.format(path, int(N)), recon_dict)
spio.savemat(path + 'data.mat', mdict)
spio.savemat('../data/SyntheticData/Phantoms/{}.mat'.format(phantomname), {'phantom': phantom})


#%%=======================================================================
# Plot data
#=========================================================================

cmin = -0.05
cmax = 0.2

UQplots.sino(b_data, p, q, path, 'sino_noisy', 'Sinogram with {} % noise'.format(rnl*100), cmin = None, cmax = None)
UQplots.image2D(phantom, n, domain, 'phantom', path, 'phantom', cmin=cmin, cmax=cmax)
UQplots.image2D(x_recon, N, domain, 'Initial Guess', path, 'x0', cmin=cmin, cmax=cmax)
