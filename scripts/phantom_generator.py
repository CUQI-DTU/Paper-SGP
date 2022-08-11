#####################################################
# Functions for generating a pipe phantom
# =================================================================
# Created by:
# Silja W Christensen @ DTU
# =================================================================
# Version 2021
# =================================================================
#####################################################

import numpy as np

def pipe_phantom(n, layers, cracks=[]):
    """
    Pipe phantom generator. 
    Generates phantom with layers consisting of concentric circles. 
    Inclution of defects optional

    Input: 
        n:      
        layers: list containing two nd array entries. Each ndarray should have the same length, corresponding to number of layers (plus zero center)
            0 radius: ndarray of radii for the layers, numbers in interval (0, 1]. 
            1 values: ndarray of layer values, numbers in interval [-1, 1].
        crakcs: list containing three entries. Each entry should have the same length, corresponding to number of cracks
            0 pos: ndarray, [[r1, theta1], [r2, theta2], ... ] containing polar coordinates of crack position
            1 len: ndarray, containing lengths of cracks (todo, check that length is within image)
            2 orient: list of strings, "radial" or "angular"

    Output: 
        nxn array representing an image containing the pipe phantom
    """

    # LAYERS SECTION

    #unpack
    radius = layers[0]
    value = layers[1]

    # Make sure we can work with the inputs
    if len(np.shape(radius))>1:
        radius = radius.flatten()
    if len(np.shape(value))>1:
        value = value.flatten()
    if isinstance(radius,(np.ndarray))== False:
        raise TypeError('The radius input must be of type ndarray')
    if isinstance(value,(np.ndarray))== False:
        raise TypeError('The value input must be of type ndarray')
    if radius.shape != value.shape:
        raise Exception('The must be the same amount of radii as values')
    if all(0 < i <= 1 for i in radius)==False:
        raise ValueError('All radii must be in the interval (0,1]')
    #if all(0 <= i <= 1 for i in value)==False:
    #        raise ValueError('All phantom values must be in the interval [0,1]')

    # init image
    phantom = np.zeros((n,n))

    # number of layers
    N = np.shape(radius)[0]
    
    # sort so we start with the largest circle
    idx = np.argsort(radius)

    # create phantom
    for i in idx[::-1]:
        # Mask
        c = np.linspace(1,-1,n, endpoint=True)
        [xx, yy] = np.meshgrid(c,c)
        mask = (xx**2 + yy**2 <= radius[i]**2)

        phantom[mask] = value[i]
    print("You have added %d layers to the pipe phantom" %N)

    # CRACKS SECTION

    if cracks:
        # unpack
        pos = cracks[0]
        leng = cracks[1]
        orient =  cracks[2]

        if isinstance(pos,(np.ndarray))== False:
            raise TypeError('The crack positions must be of type ndarray')
        if isinstance(leng,(np.ndarray))== False:
            raise TypeError('The crack lengths must be of type ndarray')
        
        # number of cracks
        NC = np.shape(leng)[0]

        for i in range(NC):
            
            # direction of crack
            if orient[i] == "angular":
                d1 = np.sin(pos[i,0])
                d2 = np.cos(pos[i,0])
            elif orient[i] == "radial":
                d1 = np.cos(pos[i,0])
                d2 = -np.sin(pos[i,0])
            else:
                print("Orientation must be radial or angular")
            
            # center of crack
            x = np.cos(pos[i,0]) * pos[i,1]
            y = -np.sin(pos[i,0]) * pos[i,1]

            # get line
            start_point = (x-d1*leng[i]/2, y-d2*leng[i]/2)
            end_point   = (x+d1*leng[i]/2, y+d2*leng[i]/2)
            # translate to pixel no. 
            start_point = np.floor(float(n)/2. * (np.array(start_point) + 1.))
            end_point   = np.floor(float(n)/2. * (np.array(end_point) + 1.))
            #start_point[0]  = n-start_point[0]
            #end_point[0]    = n-end_point[0]
            line_points = get_line(start_point, end_point)

            # Mask
            for k in range(np.shape(line_points)[0]):
                phantom[line_points[k][1], line_points[k][0]] = 0

        #print("You have added %d cracks to the pipe phantom" %NC)

    #else:
        #print("You have not added any cracks to the pipe phantom")



    return phantom
#%%
def get_line(start, end):
    """Bresenham's Line Algorithm
    Source: http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm#Python
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points
# %%

def check(p1, p2, base_array):
    """
    Source: https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array
