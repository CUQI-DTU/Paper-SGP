#####################################################
# Plotting posterior statistics and diagnostics
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2021-06
# =================================================================
#####################################################
#%%
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import colors
import scipy.stats as sps
import random
import matplotlib.animation as animation

colmap = 'gray'

#%% =================================================================
# Plot autocorrelation times
# =================================================================
def iact(chainno,tau, filename, path):
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(3)
    plt.plot(chainno,tau, '.')
    plt.xlabel('Pixel no.')
    plt.ylabel('IACT')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + filename + '.png')

#%% =================================================================
# Plot autocorrelation functions
# =================================================================
def acf(acf, tau, chainno, path):

    xmax = np.max([1.5*np.max(tau), 10])

    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(121)
    cs = ax1.plot(acf[:, 0], linewidth=2, label = 'iact = {:0.1f}'.format(tau[0]))
    ax1.legend()
    ax1.set_xlim((0, xmax))
    ax1.set_title('Autocorrelation function for pixel {:d}'.format(chainno[0]))


    ax2 = plt.subplot(122)
    cs = ax2.plot(acf[:, 1], linewidth=2, label = 'iact = {:0.1f}'.format(tau[1]))
    ax2.legend()
    ax2.set_xlim((0, xmax))
    ax2.set_title('Autocorrelation function for pixel {:d}'.format(chainno[1]))

    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'acf.png')

    f = plt.figure()
    f.set_figwidth(4)
    f.set_figheight(2)
    plt.plot(acf[:, 0], linewidth=2, label = 'iact = {:0.1f}'.format(tau[0]))
    plt.legend()
    plt.xlim((0, xmax))
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'acf1.png')

    f = plt.figure()
    f.set_figwidth(4)
    f.set_figheight(2)
    plt.plot(acf[:, 1], linewidth=2, label = 'iact = {:0.1f}'.format(tau[1]))
    plt.legend()
    plt.xlim((0, xmax))
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'acf2.png')

# =================================================================
# Plot domain
# =================================================================

def image2D(imagevec, N, domain, title, path, filename, cmin=None, cmax=None):
    fig = plt.figure(figsize=(5,4))
    fig.subplots_adjust(wspace=.5)

    ax = plt.subplot(111)
    cs = ax.imshow(imagevec.reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.05,ax.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax.set_title(title)
    ax.tick_params(axis='both', which='both', length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    plt.show()
    plt.savefig(path + filename + '.png')

# =================================================================
# Movie
# =================================================================
def imagemovie(samples, domain, N, noframes, framerate, path, filename, colmap = 'gray', cmin = None, cmax = None):
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4))

    cs = ax.imshow(samples[:,0].reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2],  aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.04,ax.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    tit = ax.set_title("Sample {0}".format(0))
    ax.tick_params(axis='both', which='both', length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    
    def update_img(i):
        tmp = samples[:,i].reshape(N,N)
        cs.set_data(tmp)
        ax.set_title("Sample {0}".format(i))
        return cs

    ani = animation.FuncAnimation(fig,update_img,frames = noframes,interval=5*1/framerate*1000)
    writer = animation.FFMpegWriter(fps=framerate)
    ani.save(path + filename + '.mp4', writer=writer)
    return ani

# =================================================================
# Plot domain on logscale
# =================================================================

def logimage2D(imagevec, N, domain, title, path, filename, cmin=None, cmax=None):
    fig = plt.figure(figsize=(5,4))
    fig.subplots_adjust(wspace=.5)

    ax = plt.subplot(111)
    cs = ax.imshow(imagevec.reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax, norm=colors.LogNorm())
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.05,ax.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax.set_title(title)
    ax.tick_params(axis='both', which='both', length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    plt.show()
    plt.savefig(path + filename + '.png')

# =================================================================
# Plot the sinogram
# =================================================================
def sino(b_data, p, q, path, filename, title, cmin = None, cmax = None):
    fig = plt.figure(figsize=(5,4))
    fig.subplots_adjust(wspace=.5)

    ax2 = plt.subplot(111)
    cs = ax2.imshow(b_data.reshape(p,q, order = 'F'), cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.02,ax2.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax2.set_title(title)

    fig.subplots_adjust(wspace=.5)
    plt.show()
    plt.savefig(path + filename + '.png')

# =================================================================
# Plot the sinogram
# =================================================================
def sino2(b_data, p, q, path, filename, cmin = None, cmax = None):
    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    im = ax.imshow(b_data.reshape(p,q, order = 'F'), aspect='auto', cmap=colmap, vmin = cmin, vmax = cmax)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set_xlabel('Projection angle [deg]')
    ax.set_ylabel('Detector pixel')
    ax.set_xticks(np.linspace(0,q,5, endpoint=True))
    plt.tight_layout()
    plt.show()
    plt.savefig(path + filename + '.png')

# =================================================================
# Prior mask
# =================================================================
def priormask(mask, N, radii, cmin, cmax, path):

    angle = np.linspace( 0 , 2 * np.pi , 150 ) 

    fig = plt.figure(figsize=(5,4))
    ax1 = fig.add_subplot(111)
    cs = ax1.imshow(mask.reshape((N,N), order = 'F'), extent=[-1, 1, -1, 1], aspect='equal', cmap='Set2', vmin = cmin, vmax = cmax, interpolation = 'none')
    cax = fig.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.05,ax1.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax1.set_title('Prior mask')
    ax1.tick_params(axis='both', which='both', length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.show()
    plt.savefig(path + 'priormask.png')


# =================================================================
# Plot 1D slices of reconstruction
# =================================================================
def slices1D(x_mean, xql, xqu, x_true, N, domain, slice_vertical, slice_horizontal, realdata, path, cmin=None, cmax=None, chainno = None):
    
    if chainno is not None:
        # compute pixel positions
        tmp = np.zeros(N**2)
        pixel_x = np.zeros(len(chainno))
        pixel_y = np.zeros(len(chainno))
        for i in range(len(chainno)):
            tmp[chainno[i]] = chainno[i]
        tmp = tmp.reshape(N,N)
        for i in range(len(chainno)):
            pixel = np.where(tmp == chainno[i])
            pixel_x[i] = pixel[0]/N*domain-domain/2
            pixel_y[i] = pixel[1]/N*domain-domain/2

    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(2, 2,
                        width_ratios=[1, 2]
                        )
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0 ,1])
    ax3 = plt.subplot(gs[1, 1])
    fig.subplots_adjust(wspace=.4)
    fig.subplots_adjust(hspace=.7)

    cs = ax1.imshow(x_mean.reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.02,ax1.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax1.axvline(x=slice_vertical,color='red')
    ax1.axhline(y=slice_horizontal,color='red')
    if chainno is not None:
        for i in range(len(chainno)):
            ax1.plot(pixel_x[i], pixel_y[i], 'go')
    ax1.set_title('Mean of posterior')
    ax1.set_xticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax1.set_yticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax1.tick_params(axis='x', rotation=30)


    slice_horizontal_idx = int(N/2-slice_horizontal/domain*N)
    y = x_mean.reshape(N,N)[slice_horizontal_idx,:]
    cilow = xql.reshape(N,N)[slice_horizontal_idx,:]
    cihigh = xqu.reshape(N,N)[slice_horizontal_idx,:]
    ax2.fill_between(np.linspace(-domain/2, domain/2, N), cilow, cihigh, color='tab:blue', alpha=.1, label = '95 % CI')
    if x_true is not None:
        y_true = x_true.reshape(N,N)[slice_horizontal_idx,:]
        ax2.plot(np.linspace(-domain/2, domain/2, N),y_true, color = 'tab:orange', label = 'true')
    ax2.plot(np.linspace(-domain/2, domain/2, N),y, color = 'tab:blue', label = 'mean')
    ax2.set_title('Horizontal')
    ax2.set_xticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend(loc = 'upper left') 
    
    slice_vertical_idx = int(N/2-slice_vertical/domain*N)
    y = x_mean.reshape(N,N)[:,slice_vertical_idx]
    cilow = xql.reshape(N,N)[:,slice_vertical_idx]
    cihigh = xqu.reshape(N,N)[:,slice_vertical_idx]
    ax3.fill_between(np.linspace(-domain/2, domain/2, N), cilow[::-1], cihigh[::-1], color='tab:blue', alpha=.1, label = '95 % CI')
    if x_true is not None:
        y_true = x_true.reshape(N,N)[:,slice_vertical_idx]
        ax3.plot(np.linspace(-domain/2, domain/2, N),y_true[::-1], color = 'tab:orange', label = 'true')
    ax3.plot(np.linspace(-domain/2, domain/2, N),y[::-1], color = 'tab:blue', label = 'mean')
    ax3.set_title('Vertical')
    ax3.set_xticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax3.tick_params(axis='x', rotation=30)

    plt.show()
    plt.savefig(path +'recon_1Dslice.png')

def slices1Dvs2(x_mean, xql, xqu, x_true, N, domain, slice_vertical, slice_horizontal, realdata, path, cmin=None, cmax=None, chainno = None):
    
    if chainno is not None:
        # compute pixel positions
        tmp = np.zeros(N**2)
        tmp[chainno[0]] = chainno[0]
        tmp[chainno[1]] = chainno[1]
        tmp = tmp.reshape(N,N)
        pixel0 = np.where(tmp == chainno[0])
        pixel0_x = pixel0[0]/N*domain-domain/2
        pixel0_y = pixel0[1]/N*domain-domain/2
        pixel1 = np.where(tmp == chainno[1])
        pixel1_x = pixel1[0]/N*domain-domain/2
        pixel1_y = pixel1[1]/N*domain-domain/2

    fig = plt.figure(figsize=(9,5))
    gs = gridspec.GridSpec(2, 2,
                        height_ratios=[2.2, 1]
                        )
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1 ,0])
    ax3 = plt.subplot(gs[1, 1])
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=.5)

    cs = ax0.imshow(x_mean.reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax0.get_position().x1+0.01,ax0.get_position().y0,0.02,ax0.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax0.axvline(x=slice_vertical,color='red')
    ax0.axhline(y=slice_horizontal,color='red')
    if chainno is not None:
        ax0.plot(pixel0_x, pixel0_y, 'go')
        ax0.plot(pixel1_x, pixel1_y, 'go')
    ax0.set_title('Mean of posterior')
    ax0.set_xticks(np.linspace(-20, 20, 5, endpoint=True)) 
    ax0.set_yticks(np.linspace(-20, 20, 5, endpoint=True)) 
    ax0.tick_params(axis='x', rotation=30)
    ax0.annotate('1.',
            xy=(0.09, 0.16), xycoords='axes fraction',
            xytext=(-1, 1), textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='bottom', color = 'r')
    ax0.annotate('2.',
            xy=(0.48, 0.91), xycoords='axes fraction',
            xytext=(-1, 1), textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='bottom', color = 'r')

    cs = ax1.imshow((xqu-xql).reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2], aspect='equal', cmap=colmap)
    cax = fig.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.02,ax1.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax1.axvline(x=slice_vertical,color='red')
    ax1.axhline(y=slice_horizontal,color='red')
    if chainno is not None:
        ax1.plot(pixel0_x, pixel0_y, 'go')
        ax1.plot(pixel1_x, pixel1_y, 'go')
    ax1.set_title("95 % interquantile \nrange of posterior")
    ax1.set_xticks(np.linspace(-20, 20, 5, endpoint=True)) 
    ax1.set_yticks(np.linspace(-20, 20, 5, endpoint=True)) 
    ax1.tick_params(axis='x', rotation=30)
    ax1.annotate('1.',
            xy=(0.09, 0.16), xycoords='axes fraction',
            xytext=(-1, 1), textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='bottom', color = 'r')
    ax1.annotate('2.',
            xy=(0.48, 0.91), xycoords='axes fraction',
            xytext=(-1, 1), textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='bottom', color = 'r')


    slice_horizontal_idx = int(N/2-slice_horizontal/domain*N)
    y = x_mean.reshape(N,N)[slice_horizontal_idx,:]
    cilow = xql.reshape(N,N)[slice_horizontal_idx,:]
    cihigh = xqu.reshape(N,N)[slice_horizontal_idx,:]
    ax2.fill_between(np.linspace(-domain/2, domain/2, N), cilow, cihigh, color='tab:blue', alpha=.1, label = '95 % CI')
    if x_true is not None:
        y_true = x_true.reshape(N,N)[slice_horizontal_idx,:]
        ax2.plot(np.linspace(-domain/2, domain/2, N),y_true, color = 'tab:orange', label = 'true')
    ax2.plot(np.linspace(-domain/2, domain/2, N),y, color = 'tab:blue', label = 'mean')
    ax2.set_aspect(100)
    ax2.set_ylim((-0.05,0.2))
    ax2.set_title('1. Horizontal')
    ax2.set_xticks(np.linspace(-20, 20, 5, endpoint=True)) 
    ax2.tick_params(axis='x', rotation=30)
    
    slice_vertical_idx = int(N/2-slice_vertical/domain*N)
    y = x_mean.reshape(N,N)[:,slice_vertical_idx]
    cilow = xql.reshape(N,N)[:,slice_vertical_idx]
    cihigh = xqu.reshape(N,N)[:,slice_vertical_idx]
    ax3.fill_between(np.linspace(-domain/2, domain/2, N), cilow[::-1], cihigh[::-1], color='tab:blue', alpha=.1, label = '95 % CI')
    if x_true is not None:
        y_true = x_true.reshape(N,N)[:,slice_vertical_idx]
        ax3.plot(np.linspace(-domain/2, domain/2, N),y_true[::-1], color = 'tab:orange', label = 'true')
    ax3.plot(np.linspace(-domain/2, domain/2, N),y[::-1], color = 'tab:blue', label = 'mean')
    ax3.set_ylim((-0.05,0.2))
    ax3.set_aspect(100)
    ax3.set_title('2. Vertical')
    ax3.set_xticks(np.linspace(-20, 20, 5, endpoint=True)) 
    ax3.tick_params(axis='x', rotation=30)

    plt.show()
    plt.savefig(path +'recon_1Dslice_vs2.png')

# =================================================================
# Plot 1D slices of reconstruction
# =================================================================
def slices1D_postreals(post_realiz, post_idx, x_mean, x_true, N, domain, slice_vertical, slice_horizontal, realdata, path, cmin=None, cmax=None):

    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(2, 2,
                        width_ratios=[1, 2]
                        )
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0 ,1])
    ax3 = plt.subplot(gs[1, 1])
    fig.subplots_adjust(wspace=.4)
    fig.subplots_adjust(hspace=.7)

    cs = ax1.imshow(x_mean.reshape(N,N), extent=[-domain/2, domain/2, -domain/2, domain/2], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.02,ax1.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    ax1.axvline(x=slice_vertical,color='red')
    ax1.axhline(y=slice_horizontal,color='red')
    ax1.set_title('Mean of posterior')
    ax1.set_xticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax1.set_yticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax1.tick_params(axis='x', rotation=30)


    slice_horizontal_idx = int(N/2-slice_horizontal/domain*N)
    if x_true is not None:
        y_true = x_true.reshape(N,N, order='F')[slice_horizontal_idx,:]
        ax2.plot(np.linspace(-domain/2, domain/2, N),y_true, color = 'k', label = 'true')
    for i in range(len(post_idx)):
        ax2.plot(np.linspace(-domain/2, domain/2, N),post_realiz[:,i].reshape(N,N).T[slice_horizontal_idx,:], linewidth = 0.3)
    ax2.set_title('Horizontal')
    ax2.set_xticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax2.tick_params(axis='x', rotation=30)
    

    slice_vertical_idx = int(N/2-slice_vertical/domain*N)
    if x_true is not None:
        y_true = x_true.reshape(N,N)[:,slice_vertical_idx]
        ax3.plot(np.linspace(-domain/2, domain/2, N),y_true[::-1], color = 'k', label = 'true')
    for i in range(len(post_idx)):
        ax3.plot(np.linspace(-domain/2, domain/2, N),post_realiz[:,i].reshape(N,N).T[slice_vertical_idx,:][::-1], linewidth = 0.3)
    ax3.set_title('Vertical')
    ax3.set_xticks(np.linspace(-domain/2, domain/2, 7, endpoint=True)) 
    ax3.tick_params(axis='x', rotation=30)
  

    plt.show()
    plt.savefig(path +'post_reals_1Dslice.png')

# =================================================================
# Plot error
# =================================================================
def error_chain(x_e, x_e_thin, path):
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(121)
    cs = ax1.plot(x_e)
    ax1.set_title('Error in x')

    ax2 = plt.subplot(122)
    cs = ax2.plot(x_e_thin)
    ax2.set_title('Error in x after burnin')

    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'error.png')

# =================================================================
# Plot hyper parameter chains
# =================================================================
def xprec_chain(x_prec, x_prec_thin, path):
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(121)
    cs = ax1.plot(x_prec)
    ax1.set_title('X precision hyper param')

    ax2 = plt.subplot(122)
    cs = ax2.plot(x_prec_thin)
    ax2.set_title('X precision hyper param after burnin')

    plt.tight_layout()
    plt.show()
    plt.savefig(path +'xprecchain.png')

# =================================================================
# Plot x chains
# =================================================================
def xchains(x_chains, x_chains_thin, chainno, tau_max, filename, path, hist = 1):

    nochains = len(chainno)
    cmap = plt.get_cmap("tab10")

    fig, axs = plt.subplots(nrows=nochains, ncols=2, figsize=(10,int(2*nochains)))
    if nochains == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(nochains):
        axs[i,0].plot(x_chains[i,:], color = cmap(i))
        axs[i,1].plot(x_chains_thin[i,:], color = cmap(i))

    axs[0,0].set_title('Pixel chain')
    axs[0,1].set_title('Pixel chain after burnin and thin with tau = %i' %(int(tau_max)))

    plt.tight_layout()
    plt.show()
    plt.savefig(path + filename + '.png')

    if hist == 1:
        fig, axs = plt.subplots(nrows=int(np.ceil(nochains/2)), ncols=2, figsize=(10,int(1*nochains+1)))
        axs = axs.flatten()
        for i in range(nochains):
            axs[i].hist(x_chains_thin[i,:], bins = 30, color = cmap(i))

        plt.tight_layout()
        plt.show()
        plt.savefig(path + filename + 'hist.png')


# =================================================================
# Plot x chains
# =================================================================
def chainhist(chains, chains_thin, filename, path):
    cmap = plt.get_cmap("tab10")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,int(2)))
    axs[0].plot(chains, color = cmap(0))
    axs[1].plot(chains_thin, color = cmap(0))
    axs[0].set_title('Pixel chain')
    axs[1].set_title('Pixel chain after burnin')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + filename + '.png')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,2))
    ax.hist(chains_thin, bins = 30, color = cmap(0))
    plt.tight_layout()
    plt.show()
    plt.savefig(path + filename + 'hist.png')

# =================================================================
# Mark chain pixels in image
# =================================================================
def markchainpixels(x_mean, x_std, chainno, N, realdata, cmin, cmax, path):

    fig = plt.figure(figsize=(5,4))
    ax2 = plt.subplot(111)
    cs = ax2.imshow(x_mean.reshape(N,N), extent=[0, 1, 0, 1], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.05,ax2.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    for i in range(len(chainno)):
        ax2.plot(pixel_x[i], pixel_y[i], 'o')
    ax2.set_title('Mean of posterior')
    ax2.tick_params(axis='both', which='both', length=0)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    if realdata == True:
        ax2.set_xlim(0.25, 0.75)  
        ax2.set_ylim(0.25, 0.75) 
    plt.show()
    plt.savefig(path + 'posterior_mean_marked_pixels.png')

    fig = plt.figure(figsize=(5,4))
    ax3 = plt.subplot(111)
    cs = ax3.imshow(x_std.reshape(N,N), extent=[0, 1, 0, 1], aspect='equal', cmap=colmap)
    cax = fig.add_axes([ax3.get_position().x1+0.01,ax3.get_position().y0,0.05,ax3.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    for i in range(len(chainno)):
        ax3.plot(pixel_x[i], pixel_y[i], 'o')
    ax3.set_title('Std of posterior')
    ax3.tick_params(axis='both', which='both', length=0)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    if realdata == True:
        ax3.set_xlim(0.25, 0.75)  
        ax3.set_ylim(0.25, 0.75) 
    plt.show()
    plt.savefig(path + 'posterior_std_marked_pixels.png')