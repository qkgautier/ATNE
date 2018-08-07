#!/usr/bin/env python

"""
License and copyright TBD soon

Author: Quentin Gautier
"""

from __future__ import division

import numpy as np
from scipy.stats.mstats import gmean
import pareto


def compute_scores(y, rows, margin=0):
    ndrange = np.ptp(y, axis=0) * margin
    total_sum = np.sum(y, axis=0)
    nrows = y.shape[0]
    scores = np.empty(len(rows), dtype=np.float)
    for i, r in enumerate(rows):
        scores[i] = np.sum(r * nrows - total_sum + ndrange * (nrows - 1))
    return scores


def approximate_pareto(y, epsilons=None, margin=0):
    """
    Uses pareto.py from https://github.com/matthewjwoodruff/pareto.py
    Returns the same data as prpt.
    """
    tagalongs = np.array(pareto.eps_sort(y, epsilons=epsilons, maximize_all=True, attribution=True))
    pareto_solutions = tagalongs[:, :y.shape[1]]
    pareto_idx = tagalongs[:, y.shape[1] + 1].astype(int)
    if margin > 0:
        miny = np.min(y, axis=0)
        ptp = pareto_solutions - miny
        margin = ptp * margin
        pareto_idx = range(y.shape[0])
        for s, m in zip(pareto_solutions, margin):
            pareto_idx = np.intersect1d(pareto_idx, np.where(np.any(y >= s - m, axis=1))[0], assume_unique=True)
        pareto_solutions = y[pareto_idx, :]
    pareto_scores = compute_scores(y, pareto_solutions)
    return pareto_solutions, pareto_idx, pareto_scores


def prpt(y, margin=0, not_strict=False):
    """
    Get Pareto optimal set of points in y with optional margin
    Return Pareto points, their indexes, and the "dominance" sum
    not_strict == True replaces > with >= (i.e. actual Pareto points with convex hull in the "good" direction) 
    """
    # So we can get a group that is margin-percent-close to the Pareto front
    ndrange = np.ptp(y,axis=0)
    margin = ndrange*margin
    
    kk = np.zeros(y.shape[0]) 
    score = np.zeros(y.shape[0])
    c = np.zeros(y.shape)
    bb = np.zeros(y.shape)

    jj = 0
    
    if not_strict:
        cmp = lambda x,y: x >= y
    else:
        cmp = lambda x,y: x > y
    
    for k in range(y.shape[0]):
        j = 0
        ak = y[k,:] # Get k-th output datum
        for i in range(y.shape[0]):
            if i != k:
                bb[j,:] = ak - y[i,:] + margin
                j += 1
                
        if np.all(np.any(cmp(bb[:j,:], 0),axis=1)):
            c[jj,:] = ak
            kk[jj] = k
            score[jj] = np.sum(bb[:j,:])
            jj += 1
            
    pareto = c[:jj,:]
    pareto_idx = kk[:jj].astype(int)
    score = score[:jj] 
    
    return pareto, pareto_idx, score



def getParetoOptimalDesigns(designs):
    """
    : designs: numDesigns x numObjectives
    : Return boolean mask of Pareto optimal designs
    """
    nperr = np.seterr(invalid='ignore') # (ignore NaN)
    isOptimal = np.ones(designs.shape[0], dtype = bool)
    for i, c in enumerate(designs):
        if isOptimal[i]:
            isOptimal[isOptimal] = np.any(designs[isOptimal]>c, axis=1)  # Remove dominated points
            isOptimal[i] = True
    np.seterr(**nperr)
    return isOptimal


def drs(allDesigns, sampledIndexes, accumFunc):
    """
    Calculate the distances to reference set.
    Accumulate the distances using the given function: (numpy_1d_array -> result)
    """
    
    # Get ground truth and estimated Pareto designs
    
    sampledDesigns = allDesigns[sampledIndexes]
    
    paretoGt  = allDesigns[getParetoOptimalDesigns(allDesigns)]
    paretoEst = sampledDesigns[getParetoOptimalDesigns(sampledDesigns)]


    # Calculate the ADRS
    
    # Note: The sign is flipped compared to the paper because
    #       we consider that higher is better for design outputs
    #       (in the paper, lower is better)

    distances = np.empty(len(paretoGt))
    
    for i, x_r in enumerate(paretoGt):
        distances[i] = np.min(np.maximum(np.max((x_r - paretoEst) / x_r, 1), 0.0))
    
    return accumFunc(distances)


def adrs(allDesigns, sampledIndexes):
    return drs(allDesigns, sampledIndexes, np.mean)

def mdrs(allDesigns, sampledIndexes):
    return drs(allDesigns, sampledIndexes, np.median)

def gadrs(allDesigns, sampledIndexes):
    return drs(allDesigns, sampledIndexes, gmean)

def adrs_mdrs(allDesigns, sampledIndexes):
    return drs(allDesigns, sampledIndexes, lambda x: (np.mean(x), np.median(x)))






