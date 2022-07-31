# -*- coding: utf-8 -*-
"""
Created on Sat July 30 16:10:23 2022

@author: Yuta Takahashi
"""

import os
import datetime
import sys
import matplotlib.pyplot as plt
sys.path.append("../Bleb2SeriesData/")

from Bleb2SeriesData.utils import *
from Bleb2SeriesData.b2sd import *
from Bleb2SeriesData.peaks import *




def main(path, save_dir, closed_iterations):
    os.makedirs(save_dir, exist_ok=True)
    img, img_gray = load_image(path, save_dir)
    img_thresh = binarization_and_closed(img_gray, save_dir, closed_iterations=closed_iterations)
    contours, cell_contours = find_contours(img, img_thresh, save_dir)
    x_g, y_g = find_g(img, cell_contours, save_dir)
    print(x_g, y_g)
    g = np.array([x_g, y_g])
    each_r, each_theta, all_r, all_r_mean, each_r_cell, each_theta_cell, all_cell_r, all_cell_r_mean = \
    contour_regression(contours, cell_contours, g, save_dir=save_dir, degree=40, n_conv=25)
    
    axs = theta2xy(each_r, each_theta)
    axs_cell = theta2xy(each_r_cell, each_theta_cell)
    
    plot_each_contours(img, axs, axs_cell, g, save_dir)
    
    # shape characteristics
    area = np.sum(all_r_mean)
    # peaks
    all_r_mean = np.array(all_r_mean)
    all_cell_r_mean = np.array(all_cell_r_mean)
    r_consider_baseline = all_r_mean - all_cell_r_mean
    peaks, _ = signal.find_peaks(r_consider_baseline, distance=20)
    peak_axs = peaks2ax(peaks, each_r, each_theta, total_length=len(all_r_mean)) 
    slided_rs, argmin = slide_argmin(r_consider_baseline)
    peaks, _ = signal.find_peaks(slided_rs, distance=20)
    
    bleb_area, rel_mins, peaks = get_bleb_areas(slided_rs, peaks)
    
    peaks, bleb_area = slided_again(peaks, argmin, l=len(all_r), bleb_area=bleb_area)    
    
    peak_axs = peaks2ax(peaks, each_r, each_theta, total_length=len(all_r))  

    plot_peaks_img(img, peak_axs, g, bleb_area,  area, save_dir, with_area=True, color="white")
    
    peaks, _ = signal.find_peaks(slided_rs, distance=20)
    _, _, peaks = get_bleb_areas(slided_rs, peaks)
    
    plot_series(r_consider_baseline, slided_rs, peaks, rel_mins, save_dir)
    

if __name__ == "__main__":
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    save_dir = "./fig/{}/".format(date)

    #path = 
    main(path=path, save_dir=save_dir)