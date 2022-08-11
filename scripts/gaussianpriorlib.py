#####################################################
# Mean and precisions for SGP
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2022
# =================================================================
#####################################################

import numpy as np
import scipy as sp
from phantom_generator import pipe_phantom
from phantomlib import drawPipe

def StructuralGaussianGMRF(N, domain, maskradii, maskcenterinner, maskcenterouter, maskid, mu, precWGauss, bndcond):
    
    D1, D2 = DifferenceMatrix2D(N, bndcond)

    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    
    mask = np.zeros((N,N))
    for i in range(len(maskid)):
        mask[drawPipe(N,domain,x,y,maskcenterinner[i,:],maskcenterouter[i,:],maskradii[i,0],maskradii[i,1])] = maskid[i] 

    mask = mask.flatten(order='F')

    x_prior = np.zeros(N**2)
    w = np.zeros(N**2)
    for i in range(max(maskid)):
        x_prior[mask == i+1] = mu[i]
        w[mask == i+1] = precWGauss[i]

    Wsq = sp.sparse.diags(np.sqrt(w), 0, format='csc')

    return mask, x_prior, w, Wsq, D1, D2

def IIDGauss(N, maskradii, mu, prec):

    x_prior = np.zeros(N**2) + mu
    w = prec*np.ones(N**2)
    Wsq = sp.sparse.diags(np.sqrt(w), 0, format='csc')
    mask = np.nan
    return mask, x_prior, w, Wsq


####################################
## Precision matrices
####################################

def DifferenceMatrix2D(N, bndcond):
    I = sp.sparse.identity(N, format='csc')

    # 1D finite difference matrix 
    one_vec = np.ones(N)
    diags = np.vstack([-one_vec, one_vec])
    if (bndcond == 'zero'):
        locs = [-1, 0]
        D = sp.sparse.spdiags(diags, locs, N+1, N).tocsc()
    elif (bndcond == 'periodic'):
        locs = [-1, 0]
        D = sp.sparse.spdiags(diags, locs, N+1, N).tocsc()
        D[-1, 0] = 1
        D[0, -1] = -1
    elif (bndcond == 'neumann'):
        locs = [0, 1]
        D = sp.sparse.spdiags(diags, locs, N, N).tocsc()
        D[-1, -1] = 0
    elif (bndcond == 'centered'):
        locs = [-1, 0, 1]
        diags = np.vstack([-one_vec, 2*one_vec, -one_vec])
        D = sp.sparse.spdiags(diags, locs, N, N).tocsc()
        D[-1, -1] = 1
        D[0, 0] = 1

    # 2D finite differences in each direction
    D1 = sp.sparse.kron(I, D).tocsc()
    D2 = sp.sparse.kron(D, I).tocsc()

    return D1, D2