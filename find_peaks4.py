#!/usr/bin/python3
import h5py
from skimage import measure
import glob 
import numpy as np
from argparse import ArgumentParser
from scipy.signal import correlate2d
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import measurements
from scipy.spatial import cKDTree
import pylab as plt
import sys
from joblib import Parallel, delayed
import os

def plot_pks( img, pk=None, ret_sub=False, **kwargs):
    if pk is None:
        pk,I = pk_pos3( img,**kwargs) 
    m = img[ img > 0].mean()
    s = img[img > 0].std()
    plt.imshow( img, vmax=m+5*s, vmin=m-s, cmap='viridis', aspect='equal', interpolation='nearest')
    ax = plt.gca()
    for cent in pk:
        circ = plt.Circle(xy=(cent[1], cent[0]), radius=3, ec='r', fc='none',lw=1)
        ax.add_patch(circ)
    plt.show()
    if ret_sub:
        return pk,I
    
def plot_pks_serial( img, pk, delay, ax):
    m = img[ img > 0].mean()
    s = img[img > 0].std()
    while ax.patches:
        _=ax.patches.pop()
    ax.images[0].set_data(img)
    ax.images[0].set_clim(m-s, m+5*s)
    for cent in pk:
        circ = plt.Circle(xy=(cent[1], cent[0]), 
            radius=5, ec='Deeppink', fc='none',lw=1)
        ax.add_patch(circ)
    plt.draw()
    plt.pause(delay)


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
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

def pk_pos2( img_, make_sparse=False, nsigs=7, sig_G=None, thresh=1, sz=4, min_snr=2.,
    min_conn=2, max_conn=20, filt=False, min_dist=0):
    if make_sparse:
        img = img_.copy() #[sz:-sz,sz:-sz].copy()
        m = img[ img > 0].mean()
        s = img[ img > 0].std()
        img[ img < m + nsigs*s] = 0
        if sig_G is not None:
            img = gaussian_filter( img, sig_G)
        lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img,sig_G)))
        locs = measurements.find_objects(lab_img)
        pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
        pos = [ p for p in pos if img[ p[0], p[1] ] > thresh]
        intens = [ img[ p[0], p[1]] for p in pos ]
    else:
        img = img_.copy()#[sz:-sz,sz:-sz].copy()
        if sig_G is not None:
            lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img,sig_G)))
        else:
            lab_img, nlab = measurements.label(detect_peaks(img))
        locs = measurements.find_objects(lab_img)
        pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
        pos =  [ p for p in pos if img[ p[0], p[1] ] > thresh]
        intens = [ img[ p[0], p[1]] for p in pos ]
    #pos = np.array(pos)+sz
    
    npeaks =len(pos)
    if min_dist and npeaks>1:
        y,x = list(map( np.array, list(zip(*pos))))
        K = cKDTree(pos)        
        XX = x.copy()
        II = np.array(intens)
        YY = y.copy()
        vals = list(K.query_pairs(min_dist))
        while vals:
            inds = [ v[ np.argmax( II[list(v)] )] for v in vals]
            inds = np.unique([ i for i in range(len(II)) if i not in np.unique(inds)])
            K = cKDTree( list(zip(XX[inds],YY[inds]))  )
            vals = list( K.query_pairs(min_dist))
            XX = XX[inds]
            YY = YY[inds]
            II = II[inds]
        pos = list(zip( YY,XX) )
        intens = II

    if filt:
        new_pos = []
        new_intens =  []
        subs = []
        for (j,i),I in zip(pos, intens):
            sub_im = img_[j-sz:j+sz+1,i-sz:i+sz+1]
            if not sub_im.size:
                #print("no sub im")
                continue
            if sub_im.size< (sz*2+1)*(sz*2+1):
                #print("no sub im siz")
                #print(sub_im.shape, sz+1,2)
                continue
            pts = np.sort( sub_im[ sub_im > 0].ravel() )
            bg = np.median( pts )
            #bg = np.median( pts )
            if bg == 0:
                #print("bg is 0")
                continue
            #noise = np.std( pts-bg)
            #noise = np.median( np.sqrt(np.mean( (pts-bg)**2) ) )
            #I2 = pts[::-1][:3].mean() #np.sort(sub_im.ravel())[::-1][:5].mean() #max()
            #pts = np.sort(sub_im.ravel())[:10]
            BG = pts[:10].mean() #max()
            noise = (pts[:10]-BG).std()
            if noise == 0:
                #print("bg is 0")
                continue
            snr = (I-BG)/noise
            #if (I2-bg)/noise < min_snr:
            if snr< min_snr:
                continue
            j2,i2 = np.unravel_index( sub_im.argmax(), sub_im.shape)
            im_n = (sub_im - BG) / noise
            blob = measure.label(im_n > min_snr)
            lab = blob[ j2, i2 ]
            connectivity = np.sum( blob == lab)
            if connectivity < min_conn:
                #print("too small")
                continue
            if connectivity > max_conn:
                #print("too big")
                continue
            new_pos.append( (j-sz+j2,i-sz+i2))
            new_intens.append( I2)
            subs.append( sub_im)
        pos = new_pos
        intens = new_intens
    if filt:
        return pos, intens
    else:
        return pos, intens



def pk_pos3( img_, make_sparse=True, nsigs=7, sig_G=None, thresh=1, sz=4, min_snr=2.,
    min_conn=2, max_conn=20, filt=False, min_dist=0, r_in=None, r_out=None, 
    cent=None, subs=False, R=None, rbins=None, run_rad_med=False):
    img = img_.copy() 
    if run_rad_med:
        img = rad_med( img, R, rbins, nsigs)
    else:
        m = img[ img > 0].mean()
        s = img[ img > 0].std()
        img[ img < m + nsigs*s] = 0
    
    if sig_G is not None:
        img = gaussian_filter( img, sig_G)
    lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img,sig_G)))
    locs = measurements.find_objects(lab_img)
    pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
    pos = [ p for p in pos if img[ p[0], p[1] ] > thresh]
    intens = [ img[ p[0], p[1]] for p in pos ]

    if r_in is not None or r_out is not None:
        assert( cent is not None)
        y = np.array([ p[0] for p in pos ]).astype(float)
        x = np.array([ p[1] for p in pos ]).astype(float)
        r = np.sqrt( (y-cent[1])**2 + (x-cent[0])**2)
        
        if r_in is not None:
            inds_in = np.where( r < r_in )[0]
        else:
            inds_in = np.array([])
        if r_out is not None:
            inds_out = np.where( r > r_out)[0]
        else:
            inds_out = np.array([])
        
        inds = np.unique( np.hstack((inds_in, inds_out))).astype(int)
        if inds.size:
            pos = [pos[i] for i in inds]
            intens = [intens[i] for i in inds]  
        else:
            return [],[]
    npeaks =len(pos)
    if min_dist and npeaks>1:
        YXI = np.hstack( (pos, np.array(intens)[:,None]))
        K = cKDTree( YXI[:,:2])        
        pairs = np.array(list( K.query_pairs( min_dist)))
        while pairs.size:
            smaller = YXI[:,2][pairs].argmin(1)
            inds = np.unique( [ pairs[i][l] for i,l in enumerate(smaller)] )
            YXI = np.delete(YXI, inds, axis=0)
            K = cKDTree( YXI[:,:2]  )
            pairs = np.array(list( K.query_pairs( min_dist)))

        pos = YXI[:,:2]
        intens = YXI[:,2]

    if filt:
        new_pos = []
        new_intens =  []
        subs = []
        for (j,i),I in zip(pos, intens):
            sub_im = img_[j-sz:j+sz+1,i-sz:i+sz+1]
            if not sub_im.size:
                continue
            if sub_im.size< (sz*2+1)*(sz*2+1):
                continue
            pts = np.sort( sub_im[ sub_im > 0].ravel() )
            bg = np.median( pts )
            if bg == 0:
                continue
            
            #I2 = pts[::-1][:2] #np.sort(sub_im.ravel())[::-1][:5].mean() #max()
            BG = np.median(pts) #max()
            diffs = np.sqrt( (pts-BG)**2) #noise = (pts[:10]-BG).std()
            #if noise == 0:
            #    #print("bg is 0")
            #    continue
            diffs_med = np.median( diffs)
            snr = 0.6745 * (I-BG).mean() / diffs_med
            #if (I2-bg)/noise < min_snr:
            if snr< min_snr:
                continue
            
            #j2,i2 = np.unravel_index( sub_im.argmax(), sub_im.shape)
            #im_n = (sub_im - BG) / noise
            #blob = measure.label(im_n > min_snr)
            #lab = blob[ j2, i2 ]
            #connectivity = np.sum( blob == lab)
            #if connectivity < min_conn:
                #print("too small")
            #    continue
            #if connectivity > max_conn:
                #print("too big")
            #    continue
            new_pos.append( [j,i] )
            #new_pos.append( (j-sz+j2,i-sz+i2))
            new_intens.append( I )
            subs.append( sub_im)
        pos = new_pos
        intens = new_intens
    if filt:
        return pos, intens
    if subs and filt:
        return pos, intens, subs

    else:
        return pos, intens







def pk_pos( img_, make_sparse=False, nsigs=7, sig_G=None, thresh=1, min_dist=None):
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
    npeaks =len(pos)
    if min_dist and npeaks>1:
        y,x = list(map( np.array, list(zip(*pos))))
        K = cKDTree(pos)        
        XX = x.copy()
        II = np.array(intens)
        YY = y.copy()
        vals = list(K.query_pairs(min_dist))
        while vals:
            inds = [ v[ np.argmax( II[list(v)] )] for v in vals]
            inds = np.unique([ i for i in range(len(II)) if i not in np.unique(inds)])
            K = cKDTree( list(zip(XX[inds],YY[inds]))  )
            vals = list( K.query_pairs(min_dist))
            XX = XX[inds]
            YY = YY[inds]
            II = II[inds]
        pos = list(zip( YY,XX) )
        intens = II
    return pos, intens


def bin_ndarray(ndarray, new_shape):
        """
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


def make_cxi_file( hits, outname, Hsh, 
    mask=None,
    dtype=np.float32, 
    verbose=True,
    min_dist=None,
    compression=None, 
    comp_opts=None, shuffle=False, thresh=1, sig_G=1, 
    make_sparse=1, nsigs=2, min_num_pks=0, log_prefix="", log_freq=10):
    
    if cent is None:
        cent = [ Hsh[1] / 2., Hsh[0]/2.]

    all_pk = []
    all_pk_intens = []
    if mask is None:
        mask = np.ones(Hsh).astype(bool)
    with h5py.File( outname, "w") as out:
        #img_dset = out.create_dataset("images", 
        img_dset = out.create_dataset('data', 
            shape=(500000,Hsh[0], Hsh[1]),
            maxshape=(None,Hsh[0], Hsh[1] ), 
            dtype=dtype,
            chunks=(1,Hsh[0], Hsh[1]),
            compression=compression, 
            compression_opts=comp_opts,
            shuffle=shuffle)

        count = 0
        for i_h,h in enumerate(hits): 
            pk, pk_I = pk_pos( h*mask, sig_G=sig_G, 
                thresh=thresh, 
                make_sparse=make_sparse, 
                nsigs=nsigs, min_dist=None)
            
            npks = len(pk_I)
            if npks <= min_num_pks:
                continue
            all_pk.append( pk)
            all_pk_intens.append( pk_I)

            count += 1
            #img_dset.resize( (count, Hsh[0], Hsh[1]),)
            img_dset[count-1] = h
            perc = float(count) / float(i_h+1) * 100.
            if verbose:
                if i_h%log_freq==0:
                    print("%s"%log_prefix)
                    print("\tFound %d / %d hits. (%.2f %%)"%( count, i_h+1,  perc))
        if count == 0:
            return
        img_dset.resize( (count, Hsh[0], Hsh[1]))
        npeaks = [len(p) for p in all_pk]
        max_n = max(npeaks)
        pk_x = np.zeros((len(all_pk), max_n))
        pk_y = np.zeros_like(pk_x)
        pk_I = np.zeros_like(pk_x)

        for i,pk in enumerate(all_pk):
            n = len( pk)
            pk_x[i,:n] = [p[1] for p in pk ]
            pk_y[i,:n] = [p[0] for p in pk ]
            pk_I[i,:n] = all_pk_intens[i]

        npeaks = np.array( npeaks, dtype=np.float32)
        pk_x = np.array( pk_x, dtype=np.float32)
        pk_y = np.array( pk_y, dtype=np.float32)
        pk_I = np.array( pk_I, dtype=np.float32)

        #out.create_dataset( 'entry_1/result_1/nPeaks' , data=npeaks)
        #out.create_dataset( 'entry_1/result_1/peakXPosRaw', data=pk_x )
        #out.create_dataset( 'entry_1/result_1/peakYPosRaw', data=pk_y )
        #out.create_dataset( 'entry_1/result_1/peakTotalIntensity', data=pk_I )
        out.create_dataset( 'peaks/nPeaks' , data=npeaks)
        out.create_dataset( 'peaks/peakXPosRaw', data=pk_x )
        out.create_dataset( 'peaks/peakYPosRaw', data=pk_y )
        out.create_dataset( 'peaks/peakTotalIntensity', data=pk_I )

    return all_pk, all_pk_intens

# blocks
def blocks(img, N):
    sub_imgs = []
    M = int( float(img.shape[0]) / N ) 
    y= 0
    for j in range( M):
        slab = img[y:y+N]
        x = 0
        for i in range(M):
            sub_imgs.append(slab[:,x:x+N] ) 
            x += N
        y += N
    return np.array( sub_imgs ) 

def make_temp(img, pk):
    temp = np.zeros_like( img)
    for i,j in pk:
        temp[i,j] = 1
    return temp

def make_temps(sub_imgs , pks ):
    temps = zeros_like ( sub_imgs ) 
    for i_pk, pk in enumerate(pks):
        for i,j in pk:
            temps[i_pk][i,j] = 1
    return temps
   

def make_2dcorr(img=None,  nsigs=4, sig_G=1.1, thresh=1, 
    make_sparse=1, block_sz=25, mode='full', temp=None ):
    if temp is None:
        assert (img is not None)
        pk,pk_I = pk_pos( img, sig_G=sig_G, 
            thresh=thresh, make_sparse=make_sparse, nsigs=nsigs)
        temp = make_temp( img, pk)
    
    temp_s = blocks( temp, N=block_sz)
    C = np.mean( [correlate2d(T,T, mode) for T in temp_s ], axis=0 ) 
    C[ C==C.max()] = 0 # mask self correlation
    return C


def make_cxi_file2( input_name, keys, outname, Hsh, 
    mask=None,
    dtype=np.float32, 
    verbose=True,
    r_in=None,
    r_out=None,
    min_dist=None,
    ninside=15,
    noutside=15,
    filt=0,
    min_snr=2.,
    min_conn=3,
    max_conn=20,
    sz=4,
    compression=None,
    sX=[None,None],
    sY=[None,None],
    comp_opts=None, shuffle=False, thresh=1, sig_G=1,
    ax=None, 
    make_sparse=1, nsigs=2, min_num_pks=0, log_prefix="", log_freq=10):
    
    h5 = h5py.File( input_name, 'r')
    images = ( h5[k]['data'].value for k in keys)
    #if ax is not None:
    #    fig = plt.figure(1)
    #    ax = plt.gca()
    all_pk = []
    all_pk_intens = []
    if mask is None:
        mask = np.ones(Hsh).astype(bool)
    with h5py.File( outname, "w") as out:
        #img_dset = out.create_dataset("images", 
        img_dset = out.create_dataset('data', 
            shape=(500000,Hsh[0], Hsh[1]),
            maxshape=(None,Hsh[0], Hsh[1] ), 
            dtype=dtype,
            chunks=(1,Hsh[0], Hsh[1]),
            compression=compression, 
            compression_opts=comp_opts,
            shuffle=shuffle)
        
        no_hits=False
        count = 0
        for i_h,h in enumerate(images): 
            
            pk, pk_I = pk_pos2( (h*mask)[sY[0]:sY[1], sX[0]:sX[1]],
                sig_G=sig_G, 
                thresh=thresh, 
                make_sparse=make_sparse, 
                nsigs=nsigs,
                min_dist=min_dist,
                min_snr=min_snr,
                min_conn=min_conn,
                max_conn=max_conn,
                sz=sz,
                filt=filt)
            
            if r_in is not None or r_out is not None:
                y = np.array([ p[0] for p in pk ]).astype(float)
                x = np.array([ p[1] for p in pk ]).astype(float)
                r = np.sqrt( (y-717.)**2 + (x-717.)**2)
                if r_in is not None:
                    inds_in = np.where( r < r_in )[0]
                else:
                    inds_in = np.array([])
                if r_out is not None:
                    inds_out = np.where( r > r_out)[0]
                else:
                    inds_out = np.array([])
                
                inds = np.unique( np.hstack((inds_in, inds_out))).astype(int)
                if not inds.size:
                    continue
                pk = [pk[i] for i in inds]
                pk_I = [pk_I[i] for i in inds]
            
            npks = len(pk_I)
            if npks <= min_num_pks:
                continue
            if ax is not None:
                plot_pks_serial((h*mask)[sY[0]:sY[1], sX[0]:sX[1]], pk, 0.001, ax)
            
            all_pk.append(pk)
            all_pk_intens.append( pk_I)

            count += 1
            img_dset[count-1] = h
            perc = float(count) / float(i_h+1) * 100.
            if verbose:
                if i_h%log_freq==0:
                    print("%s"%log_prefix)
                    print("\tFound %d / %d hits. (%.2f %%)"%( count, i_h+1,  perc))
        if count ==0:
            no_hits=True
        else:
            img_dset.resize( (count, Hsh[0], Hsh[1]))
            npeaks = [len(p) for p in all_pk]
            max_n = max(npeaks)
            pk_x = np.zeros((len(all_pk), max_n))
            pk_y = np.zeros_like(pk_x)
            pk_I = np.zeros_like(pk_x)

            for i,pk in enumerate(all_pk):
                n = len( pk)
                pk_x[i,:n] = [p[1] for p in pk ]
                pk_y[i,:n] = [p[0] for p in pk ]
                pk_I[i,:n] = all_pk_intens[i]

            npeaks = np.array( npeaks, dtype=np.float32)
            pk_x = np.array( pk_x, dtype=np.float32)
            pk_y = np.array( pk_y, dtype=np.float32)
            pk_I = np.array( pk_I, dtype=np.float32)

            out.create_dataset( 'peaks/nPeaks' , data=npeaks)
            out.create_dataset( 'peaks/peakXPosRaw', data=pk_x )
            out.create_dataset( 'peaks/peakYPosRaw', data=pk_y )
            out.create_dataset( 'peaks/peakTotalIntensity', data=pk_I )
    if no_hits:
        os.remove(outname)
    return all_pk, all_pk_intens














def rad_med(I,R, rbins, thresh, mask_val=0):
    I1 = I.copy().ravel()
    R1 = R.ravel()

    bin_assign = np.digitize( R1, rbins)
    
    for b in np.unique(bin_assign):
        if b ==0:
            continue
        if b == len(rbins):
            continue
        inds = bin_assign==b
        pts = I1[inds]
        med = np.median( pts )
        diffs = np.sqrt( (pts-med)**2 )
        med_diff = np.median( diffs)
        Zscores = 0.6745 * diffs / med_diff
        pts[ Zscores < thresh] = 0
        I1[inds] = pts
    return I1.reshape( I.shape)
        

def make_cxi_file3( img_gen, outname, Hsh, 
    mask=None,
    dtype=np.float32, 
    verbose=True,
    compression=None,
    sX=[None,None],
    sY=[None,None],
    comp_opts=None, 
    shuffle=False,
    min_num_pks=0,
    ax=None,
    max_hits=None,
    **kwargs):
    
    
    all_pk = []
    all_pk_intens = []
    if mask is None:
        mask = np.ones(Hsh).astype(bool)
    with h5py.File( outname, "w") as out:
        img_dset = out.create_dataset('data', 
            shape=(500000,Hsh[0], Hsh[1]),
            maxshape=(None,Hsh[0], Hsh[1] ), 
            dtype=dtype,
            chunks=(1,Hsh[0], Hsh[1]),
            compression=compression, 
            compression_opts=comp_opts,
            shuffle=shuffle)
        
        no_hits=False
        count = 0
        for i_h,h in enumerate(img_gen): 
            
            pk, pk_I = pk_pos3( (h*mask), **kwargs)
            
            npks = len(pk_I)
            
            if npks <= min_num_pks:
                continue
            
            if ax is not None:
                img = (h*mask )[sY[0]:sY[1], sX[0]:sX[1]]
                plot_pks_serial(img, pk, 0.001, ax)
            
            all_pk.append(pk)
            all_pk_intens.append( pk_I)

            count += 1
            img_dset[count-1] = h
            perc = float(count) / float(i_h+1) * 100.
            if verbose:
                print("\tFound %d / %d hits. (%.2f %%)"%( count, i_h+1,  perc))
            if max_hits is not None and count >= max_hits:
                break
        if count == 0:
            print("NO HITS")
            no_hits=True
        else:
            img_dset.resize( (count, Hsh[0], Hsh[1]))
            npeaks = [len(p) for p in all_pk]
            max_n = max(npeaks)
            pk_x = np.zeros((len(all_pk), max_n))
            pk_y = np.zeros_like(pk_x)
            pk_I = np.zeros_like(pk_x)

            for i,pk in enumerate(all_pk):
                n = len( pk)
                pk_x[i,:n] = [p[1] for p in pk ]
                pk_y[i,:n] = [p[0] for p in pk ]
                pk_I[i,:n] = all_pk_intens[i]

            npeaks = np.array( npeaks, dtype=np.float32)
            pk_x = np.array( pk_x, dtype=np.float32)
            pk_y = np.array( pk_y, dtype=np.float32)
            pk_I = np.array( pk_I, dtype=np.float32)

            out.create_dataset( 'peaks/nPeaks' , data=npeaks)
            out.create_dataset( 'peaks/peakXPosRaw', data=pk_x )
            out.create_dataset( 'peaks/peakYPosRaw', data=pk_y )
            out.create_dataset( 'peaks/peakTotalIntensity', data=pk_I )
    if no_hits:
        os.remove(outname)
    return all_pk, all_pk_intens

