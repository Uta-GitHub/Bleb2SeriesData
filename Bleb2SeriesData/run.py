# -*- coding: utf-8 -*-
"""
Created on Sat July 30 16:10:23 2022

Edited on Wed August 24 11:25:56 2022

@author: Yuta Takahashi
"""

import os
import datetime
import sys
import numpy as np

import matplotlib.pyplot as plt
sys.path.append("../Bleb2SeriesData/")

from Bleb2SeriesData.utils import *
from Bleb2SeriesData.b2sd import *
from Bleb2SeriesData.peaks import *




def main(path, save_dir, closed_iterations, color="white", threshold=85, cytoplasmic_region=None, not_bleb_area=2):
    os.makedirs(save_dir, exist_ok=True)
    img, img_gray = load_image(path, save_dir)
    img_thresh = binarization_and_closed(img_gray, save_dir, closed_iterations=closed_iterations, threshold=threshold)
    contours, cell_contours = find_contours(img, img_thresh, save_dir)
    if type(cytoplasmic_region) == np.ndarray:
        cell_contours = cytoplasmic_region
        img_tmp = img.copy()
        output = cv2.drawContours(img_tmp, contours, -1, (0,0,255), 3) 
        img_tmp = img.copy()
        cell_output = cv2.drawContours(img_tmp, cell_contours, -1, (0,255,0), 3)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(121)
        ax.imshow(output)
        ax.axis('off')
        ax.set_title("Contours")
    
        ax = fig.add_subplot(122)
        ax.imshow(cell_output)
        ax.set_title("Cell Body Contours")
        plt.savefig(save_dir+"Contours_nuc_manual")
    x_g, y_g = find_g(img, cell_contours, save_dir)
    #print(x_g, y_g)
    g = np.array([x_g, y_g])
    each_r, each_theta, all_r, all_r_mean, each_r_cell, each_theta_cell, all_cell_r, all_cell_r_mean = \
    contour_regression(contours, cell_contours, g, save_dir=save_dir, degree=40, n_conv=25)#contour_interpolation(contours, cell_contours, g, save_dir=save_dir, n_conv=25)
    
    axs = theta2xy(all_r_mean, each_theta) ###
    axs_cell = theta2xy(all_cell_r_mean, each_theta_cell)###
    
    plot_each_contours(img, axs, axs_cell, g, save_dir)
    
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
    peaks, bleb_area = slided_again(peaks, argmin, l=len(all_r), bleb_area=bleb_area)  ######## !!! peaksが－
    #print((bleb_area/area*100))
    #print(peaks)
    bleb_idx = (bleb_area/ area*100) >= not_bleb_area # 一定の割合以下のものはブレブでないとみなし、除く.    
    
    peak_axs = peaks2ax(peaks, each_r, each_theta, total_length=len(all_r_mean))
    bleb_area = bleb_area[bleb_idx]
    peaks = peaks[bleb_idx]
    peak_axs = peak_axs[bleb_idx]
    plot_peaks_img(img, peak_axs, g, bleb_area,  area, save_dir, with_area=True, color=color)
    
    
    
    peaks, _ = signal.find_peaks(slided_rs, distance=20)
    slided_bleb_idx = (slided_bleb_area/ area*100) >= not_bleb_area
    peaks = peaks[slided_bleb_idx]

    
    plot_series(r_consider_baseline, slided_rs, peaks, rel_mins, save_dir)
    

if __name__ == "__main__":
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    save_dir = "./fig/{}/".format(date)

    path = input("path: ")
    main(path=path, save_dir=save_dir)