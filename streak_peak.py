
import numpy as np
import pylab as plt
import h5py
from itertools import izip
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label
import glob


from astride import Streak


class multi_h5s_img:
    def __init__(self, fnames, data_path):
        self.h5s = [ h5py.File(f,'r') for f in fnames]
        self.N = sum( [h[data_path].shape[0] for h in self.h5s])
        self.data_path = data_path
        self._make_index()
    def _make_index(self):
        self.I = {}
        count = 0
        for i,h in enumerate(self.h5s):
            N_data = h[self.data_path].shape[0]
            for j in range( N_data):
                self.I[count] = {'file_i':i, 'shot_i':j}
                count += 1
    def __getitem__(self,i):
        file_i = self.I[i]['file_i']
        shot_i = self.I[i]['shot_i']
        return self.h5s[file_i][self.data_path][shot_i]

class multi_h5s_npeaks:
    def __init__(self, fnames, peaks_path):
        self.h5s = [ h5py.File(f,'r') for f in fnames]
        self.N = sum( [h['%s/nPeaks'%peaks_path].shape[0] 
            for h in self.h5s])
        self.peaks_path = peaks_path
        self._make_index()
    def _make_index(self):
        self.I = {}
        count = 0
        for i,h in enumerate(self.h5s):
            N_data = h['%s/nPeaks'%self.peaks_path].shape[0]
            for j in range(N_data):
                self.I[count] = {'file_i':i, 'shot_i':j}
                count += 1
    def __getitem__(self,i):
        file_i = self.I[i]['file_i']
        shot_i = self.I[i]['shot_i']
        return self.h5s[file_i]['%s/nPeaks'%self.peaks_path][shot_i]

class multi_h5s_peaks:
    def __init__(self, fnames, path, peaks_path):
        self.h5s = [ h5py.File(f,'r') for f in fnames]
        self.N = sum( [h['%s/nPeaks'%peaks_path].shape[0] 
            for h in self.h5s])
        self.path = path
        self.peaks_path = peaks_path
        self._make_index()
    def _make_index(self):
        self.I = {}
        count = 0
        for i,h in enumerate(self.h5s):
            npeaks = h['%s/nPeaks'%self.peaks_path]
            N_data = npeaks.shape[0]
            for j in range( N_data):
                self.I[count] = {'file_i':i, 
                    'shot_i':j, 'npeaks': int(npeaks[j])}
                count += 1
    def __getitem__(self,i):
        file_i = self.I[i]['file_i']
        shot_i = self.I[i]['shot_i']
        npeaks = self.I[i]['npeaks']
        return self.h5s[file_i][self.path][shot_i][:npeaks]


########
# process


class SubImages:
    def __init__(self, img, y,x, sz, cent=None):

        if cent is None:
            self.cent =  (.5* img.shape[0], .5* img.shape[1] ) 
        self.img = img
        self.y = y
        self.x = x
        self.sz = sz
        self._make_sub_imgs()

    def _make_sub_imgs(self):
        ########
        yo,xo=self.cent
        sz = self.sz
        img = self.img
        y = self.y
        x = self.x
        ########
        ymax = img.shape[0]
        xmax = img.shape[1]

        #y,x = map(np.array, zip(*pk))

        self.sub_imgs = []
        bounds = {'slow':[0,0,0], 'fast':[0,0,0]}
        for i,j in izip( x,y):
            j2 = int( min( ymax,j+sz  ) )
            j1 = int( max( 0,j-sz  ) )
            
            i2 = int( min( xmax,i+sz  ) )
            i1 = int( max( 0,i-sz  ) )
            bounds['slow'] = [j1,j,j2]
            bounds['fast'] = [i1,i,i2]

            r = np.sqrt( (j-yo)**2 + (i-xo)**2)
            sub = SubImage( img[ j1:j2, i1:i2 ] , bounds, r)
            self.sub_imgs.append(sub) 
     
class SubImage:
    def __init__(self, img, bounds, radius):
        self.img = img
        self.peak = int(bounds['slow'][1]), int(bounds['fast'][1])
        self.rel_peak = int(bounds['slow'][1]-bounds['slow'][0]), \
            int(bounds['fast'][1] - bounds['fast'][0] )
        self.radius = radius
      
        Y,X = np.indices( img.shape)
        self.pix_pts = np.array( zip(X.ravel(), Y.ravel() ) )
    
    def _set_streak_mask( self, **kwargs):
        streak = Streak(self.img,
            output_path='.',
            **kwargs)
        streak.detect()
        edges = streak.streaks
        if not edges:
            print("no edges!")
            self.mask =np.ones(self.img.shape, bool  ) 
        
        verts = [ np.vstack(( edge['x'], edge['y'])).T 
                        for edge in edges]
        paths = [ plt.mpl.path.Path(v) 
                for v in verts ]
        contains = np.vstack( [ p.contains_points(self.pix_pts) 
            for p in paths ])
        mask = np.any( contains,0).reshape( self.img.shape)
        self.mask = np.logical_not(mask)

    def get_streak_mask(self, **kwargs):
        self._set_streak_mask(**kwargs)
        return self.mask

    def get_stats( self, **kwargs):
        img = self.img
        
        self._set_streak_mask(**kwargs)

        bg = img[ self.mask == 1].mean()
        sig1 = img[self.mask ==1 ].std()
        sig2 = img[ self.mask==0].std()
        noise = sqrt( sig1**2 + sig2**2 )
        return bg, noise

    def integrate_streak( self, min_conn=0, max_conn=np.inf, **kwargs):
        bg, noise = self.get_stats(**kwargs)
        regions, n = label( ~self.mask)
        residual = self.img - bg
        cent_region = regions[self.rel_peak[0], self.rel_peak[1]]
        connect = sum( regions==cent_region)
        if min_conn < connect < max_conn:
            counts = residual[ regions==cent_region].sum()
        else:
            counts = np.nan
        self.counts = counts
        self.bg = bg
        self.noise = noise
        self.peak_region = regions==cent_region
        self.N_connected = connect
        #return counts, bg , noise, residual, connect
    


for s in S.sub_imgs:


def make_sub_imgs( img, pk, sz):
    ymax = img.shape[0]
    xmax = img.shape[1]

    y,x = map(np.array, zip(*pk))

    sub_imgs = []
    for i,j in izip( x,y):
        j2 = int( min( ymax,j+sz  ) )
        j1 = int( max( 0,j-sz  ) )
        
        i2 = int( min( xmax,i+sz  ) )
        i1 = int( max( 0,i-sz  ) )
        sub_imgs.append( img[ j1:j2, i1:i2 ] )
 
    return sub_imgs


def find_edges(img, sig_G=0, **kwargs):
    streak = Streak(gaussian_filter(img,sig_G), **kwargs)
    streak.detect()
    return streak.streaks

def plot_streak( img, sig_G_lst=[0], **kwargs):
    m = median( img[ img >0])
    s = std(img[ img >0])
    imshow( img, vmax=m+4*s, vmin=m-s, cmap='viridis')
    for sig_G in sig_G_lst:
        streak = Streak(gaussian_filter(img,sig_G),
            output_path='.',
            **kwargs)
        streak.detect()
        edges = streak.streaks
        if not edges:
            return 0
        verts = [ np.vstack(( edge['x'], edge['y'])).T 
                        for edge in edges]
        paths = [ plt.mpl.path.Path(v) 
                for v in verts ]
        for p in  paths:
            patch = plt.mpl.patches.PathPatch(p, 
                facecolor='none', 
                lw=2, 
                edgecolor='Deeppink')
            ax.add_patch(patch)

def get_streak_mask( img, pix_pts, **kwargs):
    #streak = Streak(gaussian_filter(img,sig_G), 
    streak = Streak(img,output_path='.',
        **kwargs)
    streak.detect()
    edges = streak.streaks
    if not edges:
        return np.ones(img.shape, bool  )
    
    verts = [ np.vstack(( edge['x'], edge['y'])).T 
                    for edge in edges]
    paths = [ plt.mpl.path.Path(v) 
            for v in verts ]
    contains = np.vstack( [ p.contains_points(pix_pts) 
        for p in paths ])
    mask = np.any( contains,0).reshape( img_sh)

    return np.logical_not(mask)

def get_stats( img, pix_pts, **kwargs):
    mask = get_streak_mask(img, pix_pts, **kwargs)
    #assert( mask is not None)
    bg = img[ mask == 1].mean()
    sig1 = img[mask ==1 ].std()
    sig2 = img[ mask==0].std()
    noise = sqrt( sig1**2 + sig2**2 )
    return mask, bg, noise

def make_snr_img( img, pix_pts, **kwargs):
    mask, bg, noise = get_stats( img, pix_pts, **kwargs)
    snr_img = (img-bg)/noise
    return snr_img

from scipy.ndimage.measurements import label

def integrate_streak( img, pix_pts, min_conn=0, max_conn=np.inf, **kwargs):
    mask, bg, noise = get_stats( img, pix_pts, **kwargs)
    regions, n = label( ~mask)
    sub_img = img - bg
    #if n ==1:
    #counts = sub_img[ regions==1 ].sum()
    #else:
    sh = sub_img.shape
    cent_region = regions[ sh[0]/2, sh[1]/2]
    connect = sum( regions==cent_region)
    if min_conn < connect < max_conn:
        counts = sub_img[ regions==cent_region].sum()
    else:
        counts = np.nan
    return counts, bg , noise, sub_img, connect

def integrate_streak_list( imgs, pix_pts, min_conn=0, max_conn=np.inf, **kwargs):
    output = np.zeros( ( len( imgs), 4  ) )
    for i_img, img in enumerate(imgs):
        counts, bg, noise, S,conn = integrate_streak( img, pix_pts, min_conn, max_conn, **pars)
        output[i_img,0] =  counts
        output[i_img,1] =  bg
        output[i_img,2] =  noise
        output[i_img,3] =  conn
    return output


def next_subs(im,y,x, sub_sz=10):
    pk = map( np.array, zip(y, x) )
    subs = make_sub_imgs( im, pk, sub_sz )
    return subs


# load some images
#imgs = np.load('streak_igms_PINK.h5py.npy' )
#x,y,I = np.load('streak_peaks_PINK.h5py.npy')
#fnames = glob.glob("/Users/damende/mar_a2a/11bunch*.cxi")
#fnames = glob.glob("/ufo2/phyco/*.cxi")

pkimgs = multi_h5s_img( fnames, "data")
pkX = multi_h5s_peaks( fnames, "peaks/peakXPosRaw", "peaks")
pkY = multi_h5s_peaks( fnames, "peaks/peakYPosRaw", "peaks")

pars= {'min_points':0, 
    'shape_cut':2, 
    'area_cut':2, 
    'radius_dev_cut':0., 
    'connectivity_angle':10.}

subs = next_subs( pkimgs[0], pkY[0], pkX[0], sub_sz=15 ) 
Y,X = np.indices( img_sh)
pix_pts = np.array( zip(X.ravel(), Y.ravel() ) )

#all_pk = []
#for i in inds:
#    all_pk.append( map( np.array , zip( y[i], x[i] ) ) )
# len(subs) == 44

nrows=6
ncols=7
fig,axs = plt.subplots(nrows=nrows, ncols=ncols)
k = 0
for row in xrange(nrows):
    for col in xrange( ncols):
        img = subs[k]
        ax = axs[row,col]
        ax.clear()
        ax.set_title(str(k))
        ax.imshow(img, aspect='auto')
       
#       raw version
        streak = Streak(gaussian_filter(img,0), output_path='.', 
            min_points=0, 
            shape_cut=1, 
            area_cut=10, 
            radius_dev_cut=0., 
            connectivity_angle=10.)
        streak.detect()
        edges = streak.streaks
        if edges:
            verts = [ np.vstack(( edge['x'], edge['y'])).T 
                for edge in edges]
            paths = [ plt.mpl.path.Path(v) 
                for v in verts ]
            for p in  paths:
                patch = plt.mpl.patches.PathPatch(p, facecolor='none', lw=2, edgecolor='Deeppink')
                ax.add_patch(patch)

#       smoothed version
        streak = Streak(gaussian_filter(img,1.4), output_path='.', 
            min_points=0, 
            shape_cut=1, 
            area_cut=10, 
            radius_dev_cut=0., 
            connectivity_angle=10.)
        streak.detect()
        edges = streak.streaks
        if edges:
            verts = [ np.vstack(( edge['x'], edge['y'])).T 
                for edge in edges]
            paths = [ plt.mpl.path.Path(v) 
                for v in verts ]
            for p in  paths:
                patch = plt.mpl.patches.PathPatch(p, facecolor='none', lw=2, edgecolor='w')
                ax.add_patch(patch)

        ax.set_xticks([])
        ax.set_yticks([])
        k+=1


plt.show()
