#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import gaussianpriorlib

#%%
# filepath and name for CGLS erconstruction.
# obtain CGLS reconstruction and save it using the "MLreconstruction.py" script
filename = '../../FORCE/data/Data_20180911/UQout_CGLS_720angles.mat' 
mat = spio.loadmat(filename)
x_CGLS = mat['x_ML'][:,15]

domain = 55
N = 500

maskid = np.array([5,1,2,3,4,5,0,0,0,0,0])

maskradii = np.array([9,11,16,17.5,22.8])

maskcenter = np.array([[0.6,0.8],
                        [0.6,0.8],
                        [0,0.4],
                        [0,0.4],
                        [0,0.4]])

#%%
fig = plt.figure(figsize=(5,4))
fig.subplots_adjust(wspace=.5)

ax = plt.subplot(111)
cs = ax.imshow(np.reshape(x_CGLS, (N,N), order = "F"), 
                cmap='gray', 
                vmin = -0.05, 
                vmax = 0.2, 
                extent=[-domain/2, domain/2, domain/2, -domain/2], 
                aspect='equal')
for i in range(5):
    circle1 = plt.Circle((maskcenter[i,0],maskcenter[i,1]), maskradii[i], color="red", fill=False, linewidth = 1)
    ax.add_patch(circle1)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.05,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.set_title("Layer Boundaries")
ax.tick_params(axis='both', which='both', length=0)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

plt.show()
plt.savefig('../output/LayerBoundaries.png')
plt.savefig('../output/LayerBoundaries.eps', format='eps')
