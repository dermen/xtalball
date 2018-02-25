import numpy as np
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from itertools import izip
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import measurements
import pylab as plt

def plot_pks( img, pk=None, **kwargs):
    """
    Plot an image with detected peaks
    
    pk, list of peak positions, [ (y0,x0), (y1,x1) ... ]
        where y,x is slow,fast scan coordinates
    if pk is None, then peaks will be detected using pk_pos script below
    **kwargs are the arguments passed to pk_pos, e.g. thresh, nsigs, make_sparse, sig_G
    """
    
    if pk is None:
        pk,_ = pk_pos( img, **kwargs)
    assert( len(pk[0]) == 2)
    m = img[ img > 0].mean()
    s = img[img > 0].std()
    plt.figure(1)
    plt.imshow( img, vmax=m+5*s, vmin=m-s, 
        cmap='hot', 
        aspect='equal', 
        interpolation='nearest')
    ax = plt.gca()
    for cent in pk:
        circ = plt.Circle(xy=(cent[1], cent[0]), 
            radius=3, 
            ec='Deeppink', 
            fc='none',
            lw=1)
        ax.add_patch(circ)
    plt.show()
    #plt.draw()
    #plt.pause(0.001)

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    
    Borrowed this from stack overflow... 
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def pk_pos( img_, make_sparse=False, nsigs=7, sig_G=None, thresh=1):
    """
    function for detecting peaks with a little flexibility... 

    Parameters
    ==========
    img_, np.ndarray
        an image, it will be copied
    
    make_sparse, bool
        whether to threshold the image according to argument nsigs or not.
        if true, all pixels below nsigs on the mean will be set to 0

    nsigs, float
        how many standard deviations above the mean should a pixel be to be considered
        as a peak

    sig_G, float
        gaussian variance, for applying gaussian smoothing prior to peak detection 

    thresh, float

    Returns
    =======
    pos, list of tuples, 
        peak positions [(y0,x0), (y1,x1) .. ]
        where y,x corresponds to the image slow,fast scan, respeectively 
    intens, list
        the intensities of the peaks, the maximum value
    
    
    Note, if make_sparse is False, then nsigs can be ignored

    """
    if make_sparse:
        img = img_.copy()
        m = img[ img > 0].mean()
        s = img[ img > 0].std()
        img[ img < m + nsigs*s] = 0
        if sig_G is not None:
            img = gaussian_filter( img, sig_G)
        lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img,sig_G)))
        locs = measurements.find_objects(lab_img)
        pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
        pos =  [ p for p in pos if img[ p[0], p[1] ] > thresh]
        intens = [ img[ p[0], p[1]] for p in pos ] 
    else:
        if sig_G is not None:
            lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img_,sig_G)))
        else:
            lab_img, nlab = measurements.label(detect_peaks(img_))
        locs = measurements.find_objects(lab_img)
        pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
        pos =  [ p for p in pos if img_[ p[0], p[1] ] > thresh]
        intens = [ img_[ p[0], p[1]] for p in pos ] 
    return pos, intens


def bin_ndarray(ndarray, new_shape):
        """
        Borrowed this from somewhere on the interwebs...

        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.
        Number of output dimensions must match number of input dimensions.
        Example
        -------
        >>> m = np.arange(0,100,1).reshape((10,10))
        >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        >>> print(n)
        [[ 22  30  38  46  54]
         [102 110 118 126 134]
         [182 190 198 206 214]
         [262 270 278 286 294]
         [342 350 358 366 374]]
        """
        compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                       ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
                ndarray = ndarray.sum(-1*(i+1))
        return ndarray


