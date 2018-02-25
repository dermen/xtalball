# coding: utf-8

import fabio
import sys
import os
import numpy as np
import glob
import h5py
from joblib import Parallel, delayed


import find_peaks4

cxi_dir = "/home/mcfuser/dermen/cxi_files"

def get_max(  flst):
    imgs = np.array( [ fabio.open(f).data for f in flst])
    img_m = np.max( imgs,0)
    return img_m

scores_fname = '/home/mcfuser/asu_output/wedge_scores.txt' #sys.argv[1]


lines = open(scores_fname,'r').readlines()

hits = [l for l in lines if len(l.split()) >10 ]

# check if already run

#good_hits = []
#for h in hits:
#    wedge_fname = h.split()[10]
#    cxi_fname = os.path.join(cxi_dir, "%s.cxi"%wedge_fname ) 
#    if not os.path.exists(cxi_fname):
#        good_hits.append( h)
#hits = good_hits
#if not hits:
#    print( "No new indexed")
#    sys.exit()
n_jobs =  min( len( hits), 64) 

R = np.load("/home/mcfuser/dermen/R.npy")
mask = np.load('/home/mcfuser/dermen/mask_nobeamstop.npy')
cent = np.load('/home/mcfuser/dermen/cent.npy')

pk_par=  { 'make_sparse':True, 
    'nsigs':4, 
    'sig_G':.8, 
    'thresh':1, 
    'sz':12, 
    'min_snr':0.15, 
    'filt':True, 
    'min_dist':3, 
    'cent':cent, 
    'R':R, 
    'rbins':np.arange( 0,R.max(), 100), 
    'run_rad_med':True }


def main(JID, hits_):
    for h in hits_:
        wedge_ID, res, _, cell, a,b,c,al,be,ga,wedge_fname, _ = h.strip().split() 
        a = float(a)
        b = float( b)
        c = float( c)
        al = float( al)
        be = float( be)
        ga = float( ga)
        res = float( res[:-1])

        run_s = wedge_fname.split('run_')[1].split('_')[0] 
        cxi_fname = os.path.join( cxi_dir, "%s_job%d.cxi"%(wedge_fname, JID) )

        print JID, cxi_fname
        cbfs = glob.glob("/home/mcfuser/asu_output/%s/xds/data/run_%s_*.cbf"\
            %(wedge_fname,run_s))
        
        img_gen = [fabio.open(cbf).data for cbf in cbfs ]
        
        find_peaks4.make_cxi_file3(img_gen, cxi_fname, mask.shape, 
            mask=mask,  **pk_par )
        
        try:
            _ =h5py.File(cxi_fname, 'r')['peaks']
        except:
            os.remove(cxi_fname)
#

hits_split = np.array_split( hits, n_jobs)
results = Parallel( n_jobs=n_jobs)(delayed(main)(jid, hits_split[jid]) 
    for jid in xrange( n_jobs))





