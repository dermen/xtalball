# coding: utf-8

import sys
import os
import numpy as np
import pandas


scores_fname = '/home/mcfuser/asu_output/wedge_scores.txt' 

lines = open(scores_fname,'r').readlines()

hits = [l for l in lines if len(l.split()) >10 ]

wedge_ID, res, _, cell, a,b,c,al,be,ga,wedge_fname, _ = map( np.array,  zip(*[ h.strip().split()  
    for h in hits]))

wedge_ID = wedge_ID.astype(int)
a = a.astype(float)
b = b.astype(float)
c = c.astype(float)
al = al.astype(float)
be = be.astype(float)
ga = ga.astype(float)

data = {'a':a, 'b':b, 'c':c, 'alpha':al, 'beta':be, 'gamma':ga,
        'wedge_id':wedge_ID, 'wedge_fname':wedge_fname, 
        'resolution':res, 'cell':cell}
df = pandas.DataFrame(data) 
df.to_pickle("indexed_scores.pkl")
df.to_csv("indexed_scores.tsv", sep='\t', float_format="%.3f")
