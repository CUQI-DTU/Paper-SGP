#####################################################
# Library containing the different aqusition geometry configurations 
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2021-05
# =================================================================
#####################################################

import numpy as np

def Data20181128(size):
    #maxAngle    = 364               # measurement max angle
    maxAngle    = 364               # measurement max angle
    offset      = 90-37             # angular offset
    shift       = -125.3            # source offset from center
    
    stc         = 600               # source to center distance
    ctd         = 500               # center to detector distance
    dlA         = 411               # full detector length
    if size == "sparseangles":
        p   = 200               # p: number of detector pixels
        q   = 36                # q: number of projection angles
        det_full = p
    elif size == "mini":
        p   = 100               # p: number of detector pixels
        q   = 60               # q: number of projection angles
        det_full = p
    elif size == "small":
        p   = 200               # p: number of detector pixels
        q   = 360               # q: number of projection angles
        det_full = p
    elif size == "limited90":
        p   = 200               # p: number of detector pixels
        q   = 270               # q: number of projection angles
        det_full = p
        maxAngle = 270
    elif size == "limited180":
        p   = 200               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        det_full = p
        maxAngle = 180
    elif size == "full":
        p   = 510               # p: number of detector pixels
        q   = 728               # q: number of projection angles
        det_full = 512
          
    dl = dlA/det_full           # width of one detector pixel
    # view angles in rad
    theta = np.linspace(0, maxAngle/180*np.pi, q+1, endpoint=True) 
    theta = theta[:-1]
    if size == "full":
        theta = theta + offset/180*np.pi

    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u

    return p, theta, stc, ctd, shift, vectors, dl, dlA

def Data2017(size):
    maxAngle    = 360               # measurement max angle
    offset      = 0             # angular offset
    shift       = -13            # source offset from center
    stc         = 59               # source to center distance
    ctd         = 41               # center to detector distance
    det_full    = 512
    if size == "sparseangles":
        p   = 507               # p: number of detector pixels
        q   = 36                # q: number of projection angles
    if size == "sparseangles20percent":
        p   = 507               # p: number of detector pixels
        q   = 72                # q: number of projection angles
    if size == "sparseangles50percent":
        p   = 507               # p: number of detector pixels
        q   = 180                # q: number of projection angles
    elif size == "full":
        p   = 507               # p: number of detector pixels
        q   = 360               # q: number of projection angles

    dlA         = 41.1*(p/det_full)              # full detector length
    dl          = dlA/p   # length of detector element

    # view angles in rad
    theta = np.linspace(0, maxAngle, maxAngle, endpoint=False) 
    theta = theta[::int(maxAngle/q)]/180*np.pi

    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u

    return p, theta, stc, ctd, shift, vectors, dl, dlA

def Data2017_real(size):
    maxAngle    = 360               # measurement max angle
    offset      = 0             # angular offset
    shift       = -12           # source offset from center
    stc         = 59               # source to center distance
    ctd         = 41               # center to detector distance
    det_full    = 512
    if size == "sparseangles":
        p   = 507               # p: number of detector pixels
        q   = 36                # q: number of projection angles
    if size == "sparseangles20percent":
        p   = 507               # p: number of detector pixels
        q   = 72                # q: number of projection angles
    if size == "sparseangles50percent":
        p   = 507               # p: number of detector pixels
        q   = 180                # q: number of projection angles
    elif size == "full":
        p   = 507               # p: number of detector pixels
        q   = 360               # q: number of projection angles

    dlA         = 41.1*(p/det_full)              # full detector length
    dl          = dlA/p   # length of detector element

    # view angles in rad
    theta = np.linspace(0, maxAngle, 360+1, endpoint=True) 
    theta = theta[:-1]
    theta = theta[::int(360/q)]/180*np.pi
    
    mag = (stc + ctd) / stc

    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u

    return p, theta, stc, ctd, shift, vectors, dl, dlA

def Data20180911(size):
    
    offset      = 0             # angular offset
    shift       = -12.5           # source offset from center
    stc         = 60               # source to center distance
    ctd         = 50               # center to detector distance
    det_full    = 512
    startAngle  = 0
    if size == "sparseangles":
        p   = 510               # p: number of detector pixels
        q   = 36                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    if size == "sparseangles20percent":
        p   = 510               # p: number of detector pixels
        q   = 72                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    if size == "sparseangles50percent":
        p   = 510               # p: number of detector pixels
        q   = 180                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    elif size == "full":
        p   = 510               # p: number of detector pixels
        q   = 360               # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    elif size == "overfull":
        p   = 510               # p: number of detector pixels
        q   = 720               # q: number of projection angles
        maxAngle    = 364               # measurement max angle
    elif size == "limited90":
        p   = 510               # p: number of detector pixels
        q   = 90               # q: number of projection angles
        startAngle = 15
        maxAngle = 105
    elif size == "limited120":
        p   = 510               # p: number of detector pixels
        q   = 120               # q: number of projection angles
        maxAngle = 120
    elif size == "limited180":
        p   = 510               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        startAngle = 15
        maxAngle = 195
    elif size == "limited180_2":
        p   = 510               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        startAngle = 180
        maxAngle = 360

    dlA         = 41.1*(p/det_full)              # full detector length
    dl          = dlA/p   # length of detector element

    # view angles in rad
    theta = np.linspace(startAngle, maxAngle, q, endpoint=False) 
    #theta = theta[:-1]
    #theta = theta[::int(728/q)]/180*np.pi
    theta = theta/180*np.pi
    
    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u

    return p, theta, stc, ctd, shift, vectors, dl, dlA

def simple_fanbeam(size):
    maxAngle    = 364               # measurement max angle
    offset      = 90-37             # angular offset
    shift       = 0             # source offset from center
    
    stc         = 2*600               # source to center distance
    ctd         = 2*500               # center to detector distance
    dlA         = 2.5*411               # full detector length
    if size == "sparseangles":
        p   = 200               # p: number of detector pixels
        q   = 36                # q: number of projection angles
        det_full = p
    elif size == "mini":
        p   = 100               # p: number of detector pixels
        q   = 60               # q: number of projection angles
        det_full = p
    elif size == "small":
        p   = 200               # p: number of detector pixels
        q   = 360               # q: number of projection angles
        det_full = p
    elif size == "limited":
        p   = 200               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        det_full = p
        maxAngle = 180
    elif size == "full":
        p   = 510               # p: number of detector pixels
        q   = 728               # q: number of projection angles
        det_full = 512

    #dl          = dlA/p             # width of one detector pixel
    dl = dlA/det_full
    # view angles in rad
    theta = np.linspace(0, maxAngle/180*np.pi, q+1, endpoint=True) 
    theta = theta[:-1]
    if size == "full":
        theta = theta + offset/180*np.pi

    # Rotate source and detector positions
    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u
        

    return p, theta, stc, ctd, shift, vectors, dl, dlA
