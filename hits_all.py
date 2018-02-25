# coding: utf-8

import fabio
import argparse
import pylab as plt
import sys
import os
import numpy as np
import glob
import h5py
from joblib import Parallel, delayed
import time

import find_peaks4


parser = argparse.ArgumentParser(
    description='Run peak finding on a live cbf dir')
parser.add_argument(
    '-tag',
    dest='tag',
    type=str,
    required=True, help='cbf dir name, e.g. phyco5_100um')

parser.add_argument(
    '--base-dir',
    dest='base_dir',
    type=str,
    required=True, help='output base directory')

parser.add_argument(
    '-n',
    dest='n_jobs',
    type=int, default=16, \
    help='number of jobs to launch')

parser.add_argument(
    '-min-pks',
    dest='min_num_pks',
    type=int, default=6, 
    help="min num peaks per hit; default is 6")

parser.add_argument('--plot', 
    dest='plot', 
    action='store_true',
    help='whether to plot')

args = parser.parse_args()


 
tag = args.tag


#########################

plot=False
min_num_pks = args.min_num_pks # min num peaks per shot
n_jobs =  args.n_jobs  # max num jobs to parallel
cbf_dir = "/data/mcfuser/asu/LCPjet/%s/"%tag  # cbfs in here   
R = np.load("/home/mcfuser/dermen/R.npy") # pixel radius, same shape as Pilat
mask = np.load('/home/mcfuser/dermen/mask_nobeamstop.npy') # mask (bad=0, good=1), same shape as Pilat
cent = np.load('/home/mcfuser/dermen/cent.npy') # center of Pilat, fast-scan, slow-scan

# param to pass to peak finder
# could make arguments but im lazy.. semi optimized... 
pk_par=  { 'make_sparse':True, 
    'sig_G':.8,  # gaussian filter sigma (applied before local max filter)
    'thresh':1,  # local max with this value pixel (ADU) will be ignored
    'sz':12,  # sub image size used to find snr
    'min_snr':0.15,  # min snr, rough compute, optimized visually for each experiment
    'filt':True,  # whether to filter by snr and min dist
    'min_dist':3,  # min dist between pixels
    'cent':cent,  # fast scan, slow scan center of pilatus
    'R':R, # provides radius of each pixel, pre-computer... same shape as pilatus
    'rbins':np.arange( 0,R.max(), 100), # sets median filter ring limits
    'nsigs':4, # how many absolute deviations from the median should a local max be to be a peak 
    'run_rad_med':True  } # filter rings by median

print pk_par
    
def cbf_gen( cbfs):
    """
    generate cbf data , if file is finished writing...
    """
    for cbf in cbfs:
        try :
            img = fabio.open(cbf).data
            yield img
        except:
            print("Failed to open %s"%cbf)
            pass

def main(JID, cbfs, cxi_dir):
    """
    sub job for each processor
    """
    if args.plot:   # interactive plotting,  hardcoded off
        if JID%4==0:
            plt.imshow(np.random.random( mask.shape) , cmap='viridis') 
            ax = plt.gca() 
        else:
            ax= None
    else:
        ax=None

    cxi_fname = os.path.join( cxi_dir, "peakaboo_job%d.cxi"%JID ) # cxi file path
    np.savetxt( cxi_fname.replace(".cxi", ".txt"), cbfs, "%s") # save the cbf files analyzed
    imgs = cbf_gen( cbfs) # generator of imgs... (not preloaded into mem)
    
    # main juice
    find_peaks4.make_cxi_file3(imgs, cxi_fname, mask.shape, 
        mask=mask, min_num_pks=min_num_pks,  ax=ax, 
        shuffle=True,  **pk_par )
    

seen_files = [] # have we seen these cbfs before
npass=1 #how many times have we searched this folder
while 1:
    # output shit
    cxi_dir = os.path.join( args.base_dir, *[args.tag,"pass%d"%npass] )
    
    #cxi_dir = "/home/mcfuser/dermen/cxi_files/%s/pass%d"%(tag,npass) # cxi files written here
#   print outputdir
    
    if not os.path.exists(cxi_dir):
        os.makedirs(cxi_dir)
    
    cbf_files = []

    cbf_files = [ os.path.join( cbf_dir, f) for f  in os.listdir(cbf_dir)
        if f.endswith('.cbf') ]
   
    cbf_files = [ f for f in cbf_files if f not in seen_files] 
    n_cbf = len( cbf_files)
   
    print("Founmake d %d new files"%n_cbf) 

#   divide cbf files evenly for each process
    cbf_split = np.array_split( cbf_files, n_jobs)
#   parallel magic
    results = Parallel( n_jobs=min( n_cbf,n_jobs))(delayed(main)(jid, cbf_split[jid], cxi_dir) 
        for jid in xrange( n_jobs))

    seen_files += cbf_files
    npass += 1
    
    print("Snoozing for 1 minute...")
    time.sleep(60)
