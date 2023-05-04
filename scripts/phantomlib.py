#####################################################
# Library containing the different phantom configurations I have used 
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2021
# =================================================================
#####################################################

import numpy as np
from phantom_generator import pipe_phantom, create_polygon

def Pipephantom01(n):
    radii  = np.array([0.95,0.9,0.70, 0.65, 0.5])
    values = np.array([0.5,0.6,0.1, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    pos = np.array([[np.pi/4, 0.8], [np.pi*2/3,0.7]])
    leng = np.array([0.4, 0.3])
    orient =  ["angular", "radial"]
    cracks = []
    cracks.append(pos)
    cracks.append(leng)
    cracks.append(orient)

    phantom = pipe_phantom(n, layers, cracks)
    
    return phantom, radii

def Pipephantom02(n):
    radii  = np.array([0.95,0.9,0.70, 0.65, 0.5])
    values = np.array([0.8,0.6,0.1, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    pos = np.array([[np.pi/4, 0.8], [np.pi*2/3,0.7]])
    leng = np.array([0.4, 0.3])
    orient =  ["angular", "radial"]
    cracks = []
    cracks.append(pos)
    cracks.append(leng)
    cracks.append(orient)

    phantom = pipe_phantom(n, layers, cracks)
    tmp = np.zeros(np.shape(phantom))
    tmp[120:160, 120:160] = 1
    phantom = phantom + tmp

    return phantom, radii

def Pipephantom03(n):
    radii  = np.array([0.8,0.75, 0.60, 0.55, 0.4])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    pos = np.array([[np.pi, 0.65], [np.pi*2/3,0.65]])
    leng = np.array([0.4, 0.2])
    orient =  ["angular", "radial"]
    cracks = []
    cracks.append(pos)
    cracks.append(leng)
    cracks.append(orient)

    phantom = pipe_phantom(n, layers, cracks)
    
    # square mask
    c = np.linspace(1,-1,n, endpoint=True)
    [xx, yy] = np.meshgrid(c,c)
    mask = (xx <= -0.35)*(xx >= -0.5)*(yy >= 0.45 )*(yy <= 0.6)

    phantom[mask] = 1

    return phantom, radii

def Pipephantom04(n):
    radii  = np.array([0.8,0.75, 0.60, 0.58, 0.4])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    pos = np.array([[np.pi, 0.65], [np.pi*2/3,0.65]])
    leng = np.array([0.02, 0.02])
    orient =  ["angular", "radial"]
    cracks = []
    cracks.append(pos)
    cracks.append(leng)
    cracks.append(orient)

    phantom = pipe_phantom(n, layers, cracks)

    return phantom, radii

def Testphantom01(n):
    radii  = np.array([0.8,0.75, 0.47, 0.45, 0.4])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    # circle mask
    no = 12
    ang = np.linspace(0,2*np.pi, no)
    dist = 0.61
    siz = np.linspace(0.02, 0.12, no)

    c = np.linspace(1,-1,n, endpoint=True)
    [xx, yy] = np.meshgrid(c,c)
    for i in range(no):
        mask = ((xx-np.cos(ang[i])*dist)**2 + (yy-np.sin(ang[i])*dist)**2 <= siz[i]**2)
        phantom[mask] = 1

    return phantom, radii

def Testphantom02(n):
    radii  = np.array([0.8,0.75, 0.47, 0.45, 0.4])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    # circle mask
    no = 12
    ang = np.linspace(0,2*np.pi, no)
    dist = 0.61
    siz = 0.06
    val = np.linspace(0.0, 1.0, no)

    c = np.linspace(1,-1,n, endpoint=True)
    [xx, yy] = np.meshgrid(c,c)
    for i in range(no):
        mask = ((xx-np.cos(ang[i])*dist)**2 + (yy-np.sin(ang[i])*dist)**2 <= siz**2)
        phantom[mask] = val[i]

    return phantom, radii

def Testphantom03(n):
    radii  = np.array([0.8,0.75, 0.47, 0.45, 0.4])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    # circle mask
    no = 12
    ang = np.linspace(0,2*np.pi, no)
    dist = np.linspace(0.46, 0.74, no)
    siz = 0.06
    val = 1

    c = np.linspace(1,-1,n, endpoint=True)
    [xx, yy] = np.meshgrid(c,c)
    for i in range(no):
        mask = ((xx-np.cos(ang[i])*dist[i])**2 + (yy-np.sin(ang[i])*dist[i])**2 <= siz**2)
        phantom[mask] = val

    return phantom, radii

def Testphantom04(n):
    radii  = np.array([0.8,0.75, 0.47, 0.45, 0.4])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    # radial cracks
    no = 12
    nohalf = int(no/2)
    ang = np.linspace(0,2*np.pi, no, endpoint=False)
    dist = 0.61
    w = np.hstack((np.linspace(0.01, 0.05, nohalf), 0.1*np.ones(nohalf)))
    l = np.hstack((0.1*np.ones(nohalf), np.linspace(0.01, 0.05, nohalf)))
    val = 0

    for i in range(no):
        # coordinates in (x,y), -1 to 1 system
        coordinates0 = np.array([
            [w[i]/2, dist + l[i]/2],
            [-w[i]/2, dist + l[i]/2],
            [-w[i]/2, dist - l[i]/2],
            [w[i]/2, dist - l[i]/2]
        ])
        R = np.array([
            [np.cos(ang[i]), -np.sin(ang[i])],
            [np.sin(ang[i]), np.cos(ang[i])]
            ])
        coordinates = R @ coordinates0.T
        coordinates = coordinates.T

        # transform into (row, column) indicies
        vertices = np.ceil((np.fliplr(coordinates) + 1)/2*n)-1
        # create mask
        mask = create_polygon([n,n], vertices)
        mask = np.array(mask, dtype=bool)
        phantom[mask] = val

    return phantom, radii

def Testphantom04_LargeDomain(n):
    radii  = np.array([0.35,0.32, 0.24, 0.23, 0.20])
    values = np.array([0.8,0.6, 0.2, 0.7, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    # radial cracks
    no = 12
    nohalf = int(no/2)
    ang = np.linspace(0,2*np.pi, no, endpoint=False)
    dist = 0.28
    w = np.hstack((np.linspace(0.005, 0.015, nohalf), 0.04*np.ones(nohalf)))
    l = np.hstack((0.04*np.ones(nohalf), np.linspace(0.005, 0.015, nohalf)))
    val = 0

    for i in range(no):
        # coordinates in (x,y), -1 to 1 system
        coordinates0 = np.array([
            [w[i]/2, dist + l[i]/2],
            [-w[i]/2, dist + l[i]/2],
            [-w[i]/2, dist - l[i]/2],
            [w[i]/2, dist - l[i]/2]
        ])
        R = np.array([
            [np.cos(ang[i]), -np.sin(ang[i])],
            [np.sin(ang[i]), np.cos(ang[i])]
            ])
        coordinates = R @ coordinates0.T
        coordinates = coordinates.T

        # transform into (row, column) indicies
        vertices = np.ceil((np.fliplr(coordinates) + 1)/2*n)-1
        # create mask
        mask = create_polygon([n,n], vertices)
        mask = np.array(mask, dtype=bool)
        phantom[mask] = val

    return phantom, radii


def OneLayerPhantom(n):
    radii  = np.array([0.8, 0.4])
    values = np.array([0.6, 0])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    return phantom, radii

def CirclePhantom(n):
    radii  = np.array([0.8])
    values = np.array([0.6])
    layers = []
    layers.append(radii)
    layers.append(values)

    phantom = pipe_phantom(n, layers)

    return phantom, radii
#%%

def DeepSeaOilPipe(N,defects):

    radii  = np.array([2.5,3,9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    # phantom = 2e-2*8*drawPipe(N,domain,x,y,2.5,3)        # Axis  (8.05g/cm^3)
    # phantom = phantom+2e-2*8*drawPipe(N,domain,x,y,9,11)      # Steel (8.05g/cm^3)
    # phantom = phantom+1e-1*0.1*drawPipe(N,domain,x,y,11,16)      # Foam PU?  0.1-0.8 g / cm^3
    # phantom = phantom+1e-1*0.94*drawPipe(N,domain,x,y,16,17.5)     # PE      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    # phantom = phantom+5e-2*2.3*drawPipe(N,domain,x,y,17.5,23)    # Concrete 2.3 g/cm^3
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,2.5,3)        # Axis  (8.05g/cm^3)
    phantom = phantom+2e-2*7.9*drawPipe(N,domain,x,y,center,center,9,11)      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,11,16)      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,16,17.5)     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,17.5,23)    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:
        no = 6
        ang = np.linspace(0,2*np.pi, no, endpoint=False)
        dist = 20.25/domain*N
        w = 0.5/domain*N
        l = 4/domain*N
        val = 2e-2*7.9 # Steel
        defect_rot_ang = np.linspace(0,np.pi/2, no, endpoint=True)

        for i in range(no):
            # coordinates in (x,y)
            coordinates0 = np.array([
                [c[0]+w/2, c[1]+dist + l/2],
                [c[0]-w/2, c[1]+dist + l/2],
                [c[0]-w/2, c[1]+dist - l/2],
                [c[0]+w/2, c[1]+dist - l/2]
            ])
            # Rotation matrix around center
            R1 = np.array([
                [np.cos(ang[i]), -np.sin(ang[i])],
                [np.sin(ang[i]), np.cos(ang[i])]
                ])
            # Rotation matrix around defect
            R2 = np.array([
                [np.cos(defect_rot_ang[i]), -np.sin(defect_rot_ang[i])],
                [np.sin(defect_rot_ang[i]), np.cos(defect_rot_ang[i])]
                ])

            # Rotate defect
            coordinates = R2 @ (coordinates0.T - np.array([[c[0]],[c[1]+dist]])) + np.array([[c[0]],[c[1]+dist]])
            # Rotate around image center
            coordinates = R1 @ (coordinates - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices = np.ceil((np.fliplr(coordinates)))
            # create mask
            mask = create_polygon([N,N], vertices)
            mask = np.array(mask, dtype=bool)
            phantom[mask] = val

    return phantom, radii

def DeepSeaOilPipe2(N,defects):

    radii  = np.array([2.5,3,9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    # phantom = 2e-2*8*drawPipe(N,domain,x,y,2.5,3)        # Axis  (8.05g/cm^3)
    # phantom = phantom+2e-2*8*drawPipe(N,domain,x,y,9,11)      # Steel (8.05g/cm^3)
    # phantom = phantom+1e-1*0.1*drawPipe(N,domain,x,y,11,16)      # Foam PU?  0.1-0.8 g / cm^3
    # phantom = phantom+1e-1*0.94*drawPipe(N,domain,x,y,16,17.5)     # PE      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    # phantom = phantom+5e-2*2.3*drawPipe(N,domain,x,y,17.5,23)    # Concrete 2.3 g/cm^3
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,2.5,3)        # Axis  (8.05g/cm^3)
    phantom = phantom+2e-2*7.9*drawPipe(N,domain,x,y,center,center,9,11)      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,11,16)      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,16,17.5)     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,17.5,23)    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:
        no = 6
        ang1 = np.linspace(0,np.pi, no, endpoint=False)
        ang2 = np.linspace(np.pi,2*np.pi, no, endpoint=False)
        dist = 20.25/domain*N
        w1 = 0.2/domain*N
        w2 = 0.7/domain*N
        l = 4/domain*N
        val = 2e-2*7.9 # Steel
        defect_rot_ang = np.linspace(0,np.pi/2, no, endpoint=True)

        for i in range(no):
            # coordinates in (x,y)
            coordinates0 = np.array([
                [c[0]+w1/2, c[1]+dist + l/2],
                [c[0]-w1/2, c[1]+dist + l/2],
                [c[0]-w1/2, c[1]+dist - l/2],
                [c[0]+w1/2, c[1]+dist - l/2]
            ])
            # Rotation matrix around center
            R1 = np.array([
                [np.cos(ang1[i]), -np.sin(ang1[i])],
                [np.sin(ang1[i]), np.cos(ang1[i])]
                ])
            # Rotation matrix around defect
            R2 = np.array([
                [np.cos(defect_rot_ang[i]), -np.sin(defect_rot_ang[i])],
                [np.sin(defect_rot_ang[i]), np.cos(defect_rot_ang[i])]
                ])

            # Rotate defect
            coordinates = R2 @ (coordinates0.T - np.array([[c[0]],[c[1]+dist]])) + np.array([[c[0]],[c[1]+dist]])
            # Rotate around image center
            coordinates = R1 @ (coordinates - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices = np.ceil((np.fliplr(coordinates)))
            # create mask
            mask = create_polygon([N,N], vertices)
            mask = np.array(mask, dtype=bool)
            phantom[mask] = val
        
        for i in range(no):
            # coordinates in (x,y)
            coordinates0 = np.array([
                [c[0]+w2/2, c[1]+dist + l/2],
                [c[0]-w2/2, c[1]+dist + l/2],
                [c[0]-w2/2, c[1]+dist - l/2],
                [c[0]+w2/2, c[1]+dist - l/2]
            ])
            # Rotation matrix around center
            R1 = np.array([
                [np.cos(ang2[i]), -np.sin(ang2[i])],
                [np.sin(ang2[i]), np.cos(ang2[i])]
                ])
            # Rotation matrix around defect
            R2 = np.array([
                [np.cos(defect_rot_ang[i]), -np.sin(defect_rot_ang[i])],
                [np.sin(defect_rot_ang[i]), np.cos(defect_rot_ang[i])]
                ])

            # Rotate defect
            coordinates = R2 @ (coordinates0.T - np.array([[c[0]],[c[1]+dist]])) + np.array([[c[0]],[c[1]+dist]])
            # Rotate around image center
            coordinates = R1 @ (coordinates - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices = np.ceil((np.fliplr(coordinates)))
            # create mask
            mask = create_polygon([N,N], vertices)
            mask = np.array(mask, dtype=bool)
            phantom[mask] = val

    return phantom, radii

def DeepSeaOilPipe3(N,defects):

    radii  = np.array([2.5,3,9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    # phantom = 2e-2*8*drawPipe(N,domain,x,y,2.5,3)        # Axis  (8.05g/cm^3)
    # phantom = phantom+2e-2*8*drawPipe(N,domain,x,y,9,11)      # Steel (8.05g/cm^3)
    # phantom = phantom+1e-1*0.1*drawPipe(N,domain,x,y,11,16)      # Foam PU?  0.1-0.8 g / cm^3
    # phantom = phantom+1e-1*0.94*drawPipe(N,domain,x,y,16,17.5)     # PE      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    # phantom = phantom+5e-2*2.3*drawPipe(N,domain,x,y,17.5,23)    # Concrete 2.3 g/cm^3
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,2.5,3)        # Axis  (8.05g/cm^3)
    phantom = phantom+2e-2*7.9*drawPipe(N,domain,x,y,center,center,9,11)      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,11,16)      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,16,17.5)     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,17.5,23)    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:
        # radial cracks
        no = 12
        nohalf = int(no/2)
        ang = np.linspace(0,2*np.pi, no, endpoint=False)
        dist = 0.28
        dist = 20.25/domain*N
        w1 = 0.2/domain*N
        w2 = 0.7/domain*N
        l1 = 4/domain*N
        val = 2e-2*7.9 # Steel
        w = np.hstack((np.linspace(w1, w2, nohalf), l1*np.ones(nohalf)))
        l = np.hstack((l1*np.ones(nohalf), np.linspace(w1, w2, nohalf)))

        for i in range(no):
            # coordinates in (x,y), -1 to 1 system
            coordinates0 = np.array([
                [c[0]+w[i]/2, c[1]+dist + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist - l[i]/2],
                [c[0]+w[i]/2, c[1]+dist - l[i]/2]
            ])
            R = np.array([
                [np.cos(ang[i]), -np.sin(ang[i])],
                [np.sin(ang[i]), np.cos(ang[i])]
                ])
            # Rotate around image center
            coordinates = R @ (coordinates0.T - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices = np.ceil(np.fliplr(coordinates))
            # create mask
            mask = create_polygon([N,N], vertices)
            mask = np.array(mask, dtype=bool)
            phantom[mask] = val 

    return phantom, radii

def DeepSeaOilPipe4(N,defects):

    radii  = np.array([2.5,3,9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,9,11)      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,11,16)      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,16,17.5)     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,17.5,23)    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:
        # radial cracks
        no = 12
        nohalf = int(no/2)
        ang = np.linspace(0,2*np.pi, no, endpoint=False)
        dist = 0.28
        dist = 20.25/domain*N
        w1 = 0.2/domain*N
        w2 = 0.7/domain*N
        l1 = 4/domain*N
        val = 2e-2*7.9 # Steel
        w = np.hstack((np.linspace(w1, w2, nohalf), l1*np.ones(nohalf)))
        l = np.hstack((l1*np.ones(nohalf), np.linspace(w1, w2, nohalf)))

        for i in range(no):
            # coordinates in (x,y), -1 to 1 system
            coordinates0 = np.array([
                [c[0]+w[i]/2, c[1]+dist + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist - l[i]/2],
                [c[0]+w[i]/2, c[1]+dist - l[i]/2]
            ])
            R = np.array([
                [np.cos(ang[i]), -np.sin(ang[i])],
                [np.sin(ang[i]), np.cos(ang[i])]
                ])
            # Rotate around image center
            coordinates = R @ (coordinates0.T - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices = np.ceil(np.fliplr(coordinates))
            # create mask
            mask = create_polygon([N,N], vertices)
            mask = np.array(mask, dtype=bool)
            phantom[mask] = val 

    return phantom, radii


def DeepSeaOilPipe8(N,defects):

    radii  = np.array([9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,radii[0],radii[1])      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,radii[1],radii[2])      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,radii[2],radii[3])     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,radii[3],radii[4])    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:

        defectmask = []
        vertices = []

        # radial and angular cracks
        no = 12
        ang = np.array([-3*np.pi/9, -2*np.pi/9, -np.pi/9, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2, 2*np.pi/3, 5*np.pi/4-np.pi/9, 5*np.pi/4, 5*np.pi/4+np.pi/9])-60/180*np.pi
        dist = np.array([20.25, 20.25, 20.25, 20.25, 20.25, 16.75, 13.5, 10, 20.25, 16.75+2, 16.75, 16.75-2])/domain*N
        w = np.array([0.5, 0.4, 0.3, 0.2, 4, 4, 4, 4, 0.4, 0.4, 0.4, 0.4])/domain*N
        l = np.array([4, 4, 4, 4, 0.4, 0.4, 0.4, 0.4, 4, 4, 4, 4])/domain*N
        vals = np.zeros(no)
        vals[8] = 2e-2*7.9
        for i in range(no):
            # coordinates in (x,y), -1 to 1 system
            coordinates0 = np.array([
                [c[0]+w[i]/2, c[1]+dist[i] + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist[i] + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist[i] - l[i]/2],
                [c[0]+w[i]/2, c[1]+dist[i] - l[i]/2]
            ])
            R = np.array([
                [np.cos(ang[i]), -np.sin(ang[i])],
                [np.sin(ang[i]), np.cos(ang[i])]
                ])
            # Rotate around image center
            coordinates = R @ (coordinates0.T - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices.append(np.ceil(np.fliplr(coordinates)))
            # create mask
            tmpmask = create_polygon([N,N], vertices[i])
            defectmask.append(np.array(tmpmask, dtype=bool))
            phantom[defectmask[i]] = vals[i]

        # Cross
        c_cross_ang = -np.pi/2
        c_cross_dist = 20.25/domain*N
        c_cross = c_cross_dist*np.array([np.cos(c_cross_ang), np.sin(c_cross_ang)])+N/2
        a = (2/np.sqrt(2))/domain*N
        b = (0.2/np.sqrt(2))/domain*N
        coordinates_cross1 = np.array([
            [c_cross[0]-a+b, c_cross[1]-a],
            [c_cross[0]+a, c_cross[1]+a-b],
            [c_cross[0]+a-b, c_cross[1]+a],
            [c_cross[0]-a, c_cross[1]-a+b]])
        coordinates_cross2 = np.array([
            [c_cross[0]+a-b, c_cross[1]-a],
            [c_cross[0]+a, c_cross[1]-a+b],
            [c_cross[0]-a+b, c_cross[1]+a],
            [c_cross[0]-a, c_cross[1]+a-b]])
        # transform into (row, column) indicies
        vertices.append(np.ceil(np.flipud(coordinates_cross1)))
        # create mask
        tmpmask = create_polygon([N,N], vertices[12])
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[12]] = 0
        # transform into (row, column) indicies
        vertices.append(np.ceil(np.flipud(coordinates_cross2)))
        # create mask
        tmpmask = create_polygon([N,N], vertices[13])
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[13]] = 0

        # Circles
        ang_circ = np.array([3*np.pi/4+np.pi/9, 3*np.pi/4+np.pi/9, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4-np.pi/9])-60/180*np.pi
        dist_circ = 20.25/domain*N
        siz = np.array([1, 0.3, 1, 0.3, 0.3])/domain*N
        val = np.array([0, 2e-2*7.9, 0, 4.56e-2*2.3, 2e-2*7.9])

        for i in range(len(ang_circ)):
            tmpmask = ((x-np.cos(ang_circ[i])*dist_circ)**2 + (y-np.sin(ang_circ[i])*dist_circ)**2 <= siz[i]**2)
            defectmask.append(np.array(tmpmask, dtype=bool))
            phantom[defectmask[14+i]] = val[i]

        center_dists = np.hstack([dist, c_cross_dist, dist_circ*np.ones(3)])
        center_x = center_dists*np.hstack([np.sin(-ang), np.sin(np.array([c_cross_ang])), np.cos(ang_circ[np.array([0,2,4])])])+N/2
        center_y = center_dists*np.hstack([np.cos(-ang), np.cos(np.array([c_cross_ang])), np.sin(ang_circ[np.array([0,2,4])])])+N/2
        centers = np.vstack([center_x, center_y])
        
        return phantom, radii, defectmask, vertices, centers
    else:
        return phantom, radii


def drawPipe(N, domain, x,y ,c1,c2, r1, r2):
    # N is number of pixels on one axis
    # domain is true size of one axis
    # x and y is a meshgrid of the domain
    # r1 and r2 are the inner and outer radii of the pipe layer
    R1 = r1/domain*N
    R2 = r2/domain*N

    M1 = (x-c1[0]/domain*N)**2+(y-c1[1]/domain*N)**2>=R1**2
    M2 = (x-c2[0]/domain*N)**2+(y-c2[1]/domain*N)**2<=R2**2

    return np.logical_and(M1, M2)


# %%
