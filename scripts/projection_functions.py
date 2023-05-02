# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2020-06
# ========================================================================
import numpy as np
# from scipy.sparse import linalg as splinalg
import astra
import geometrylib

#%%=======================================================================
# geometry parameters
#=========================================================================
# Aqusition geometry
p, theta, stc, ctd, shift, vectors, dl, dlA = geometrylib.Data20180911("sparseangles20percent")

q = np.shape(vectors)[0]
# Reconstruction geometry
domain      = 55              # physical size of object
N = 500

#%%=======================================================================
# setup ASTRA
#=========================================================================

# geometries
vol_geom = astra.create_vol_geom(N,N,-domain/2,domain/2,-domain/2,domain/2)
proj_geom = astra.create_proj_geom('fanflat_vec', p, vectors)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom) # line_fanflat

print("Astra has been setup")

# =================================================================
# forward model
# ================================================================= 
def A(x, flag):
    if flag == 1:
        # forward projection
        return proj_forward_sino(x)
    elif flag == 2:
        # backward projection  
         return proj_backward_sino(x)

#=========================================================================
def proj_forward_sino(x):     
    # forward projection
    id, Ax = astra.create_sino(x.reshape((N,N), order='F'), proj_id)
    astra.data2d.delete(id)
    return Ax.flatten()

#=========================================================================
def proj_backward_sino(b):          
    # backward projection   
    b = b.reshape((q, p)) # angles,det
    id, ATb = astra.create_backprojection(b, proj_id)
    astra.data2d.delete(id)
    return ATb.flatten(order='F')