# -*- coding: utf-8 -*-
"""
Created on Wed August 24 12:09:12 2022

@author: Yuta Takahashi
"""

import sys
import numpy as np
import pandas as pd
sys.path.append("../Bleb2SeriesData/")
import cv2

from Bleb2SeriesData.utils import *
from Bleb2SeriesData.b2sd import *
from Bleb2SeriesData.peaks import *

import matplotlib.pyplot as plt#


def bleb_num_size(img_thresh, cyto_img, cluster_size=[5, 10, 15], not_bleb_area=2, plot=False):
    
    """
    img_thresh : binary image of bleb cell.
    cyto_img : image manually drawn cytoplasmic region.
    
    """
    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_RGB2GRAY)
    contours, _ = find_contours(img_thresh, img_thresh, "", plot=False)
    
    # 細胞質領域抽出
    cyto_img = cv2.cvtColor(cyto_img, cv2.COLOR_BGR2GRAY)
    _, cyto_thresh = cv2.threshold(cyto_img, 250, 255, cv2.THRESH_BINARY)
    cell_contours, _ = cv2.findContours(cyto_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_idx = np.argsort([len(arr) for arr in cell_contours])
    cell_contours = cell_contours[sorted_idx[-1]]
    
    ##
    x_g, y_g = find_g(img_thresh, cell_contours, "", plot=False)
    
    g = np.array([x_g, y_g])
    
    each_r, each_theta, all_r, all_r_mean, each_r_cell, each_theta_cell, all_cell_r, all_cell_r_mean = \
    contour_regression(contours, cell_contours, g, save_dir="", degree=40, n_conv=25, plot=False)
    
    axs = theta2xy(all_r_mean, each_theta) ###
    axs_cell = theta2xy(all_cell_r_mean, each_theta_cell)###
    
    # shape characteristics
    area = np.sum(all_cell_r_mean)
    # peaks
    all_r_mean = np.array(all_r_mean)
    all_cell_r_mean = np.array(all_cell_r_mean)
    r_consider_baseline = all_r_mean - all_cell_r_mean
    
   
    slided_rs, argmin = slide_argmin(r_consider_baseline)
    peaks, _ = signal.find_peaks(slided_rs, distance=20)
    
    bleb_area, rel_mins, peaks = get_bleb_areas(slided_rs, peaks)
    slided_bleb_area = bleb_area
    peaks, bleb_area = slided_again(peaks, argmin, l=len(all_r), bleb_area=bleb_area) 
    bleb_idx = (bleb_area/ area*100) >= not_bleb_area # 一定の割合以下のものはブレブでないとみなし、除く.    
    peak_axs = peaks2ax(peaks, each_r, each_theta, total_length=len(all_r_mean))##
    bleb_area = bleb_area[bleb_idx]
    peaks = peaks[bleb_idx]
    peak_axs = peak_axs[bleb_idx]##

    peaks, _ = signal.find_peaks(slided_rs, distance=20)
    slided_bleb_idx = (slided_bleb_area/ area*100) >= not_bleb_area
    peaks = peaks[slided_bleb_idx]  
    
    
    bleb_size_ratio = bleb_area/area*100
    # datas for cluster
    num = len(peaks)
    num_size_data = [num]
    
    c_size_min = 0
    for c_size_max in cluster_size:
        a = c_size_min <= bleb_size_ratio
        b = bleb_size_ratio < c_size_max
        num_size_data.append(np.sum(a*b)) # a かつ bがTrueの個数
        
        c_size_min = c_size_max
    num_size_data.append(np.sum(c_size_min <= bleb_size_ratio  )) # 最大クラスのサイズ
    
    if plot:
        plot_peaks_img(cyto_img, peak_axs, g, bleb_area,  area, with_area=True)
    
    return np.array(num_size_data)


def get_datas(img_thresh_paths, cyto_img_paths, cluster_size=[5, 10, 15], not_bleb_area=2, plot=False):
    s = 0
    for t_path, c_path in zip(img_thresh_paths, cyto_img_paths):
        img = cv2.imread(t_path)
        cyto = cv2.imread(c_path)
        if s == 0:
            bleb_datas = bleb_num_size(img, cyto, cluster_size, not_bleb_area=not_bleb_area, plot=plot)
        else:
            bleb_datas = \
            np.vstack((bleb_datas, bleb_num_size(img, cyto, cluster_size, not_bleb_area=not_bleb_area, plot=plot)))
        s += 1
        if s % 10==9:
            print(f"{s}枚目を処理")
        
        # column name作製
        col = ["num"]
        s0 = not_bleb_area
        for s in cluster_size:
            col.append(f"{s0}~{s}")
            s0 = s
        col.append(f"{cluster_size[-1]}~")
            
    return pd.DataFrame(bleb_datas, columns=col)
        
    