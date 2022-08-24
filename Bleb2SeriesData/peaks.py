# -*- coding: utf-8 -*-
"""
Created on Sat July 30 17:10:07 2022

@author: Yuta Takahashi
"""
import sys
sys.path.append("../Bleb2SeriesData/")

import numpy as np
import scipy.signal as signal
import scipy
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from Bleb2SeriesData.b2sd import *



def slide_argmin(r):
    argmin = np.argmin(r)
    slided_r = np.concatenate((r[argmin:], r[:argmin]))
    return slided_r, argmin

def get_bleb_areas(rs, peaks):
    rel_mins=[]
    for i in range(len(peaks)-1):
        rel_min = np.argmin(rs[peaks[i]:peaks[i+1]]) + peaks[i]
        rel_mins.append(rel_min)
        
        

    bleb_areas = [np.sum(rs[:rel_mins[0]])]
    for i in range(len(rel_mins)-1):
        area = np.sum(rs[rel_mins[i]:rel_mins[i+1]])
        bleb_areas.append(area)
        
    bleb_areas.append(np.sum(rs[rel_mins[len(rel_mins)-1]:]))

        
    return np.array(bleb_areas), rel_mins, peaks

def slided_again(peaks, argmin, l, bleb_area):
    slided_peaks = peaks + argmin
    idx = np.argmin(slided_peaks < l)
    slided_peaks = np.concatenate((slided_peaks[idx:]-l, slided_peaks[:idx]))
    slided_bleb_area = np.concatenate((bleb_area[idx:], bleb_area[:idx]))
    return slided_peaks, slided_bleb_area


def peaks2ax(peaks, each_r, each_theta, total_length=760):
    """scipy.signal関連のpeaksを座標に変換"""
    #print("peaks", peaks)
    r_1, r_2, r_3, r_4 = [], [], [], []
    t_1, t_2, t_3, t_4 = [], [], [], []
    rs = [r_1, r_2, r_3, r_4]
    thetas = [t_1, t_2, t_3, t_4]
    each_length = int(total_length/4) #各象限の長さ
    for p in peaks:
        for i in range(4):
            if i*each_length <= p < (i+1)*each_length:
                #print(i, p)
                r = each_r[i][p-i*each_length]
                theta = each_theta[i][p-i*each_length]
                rs[i].append(r)
                thetas[i].append(theta)
                
    peak_axs = old_theta2xy(rs, thetas)##
    
    return peak_axs