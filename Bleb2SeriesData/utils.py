# -*- coding: utf-8 -*-
"""
Created on Sat July 30 15:30:22 2022

@author: Yuta Takahashi
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
#import seaborn as sns
import sys
sys.path.append("../Bleb2SeriesData/")

### load images and Binarization ###

def load_image(path, save_dir):
    img = cv2.imread(path)
    original_img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fig, ax = plt.subplots(1,2, figsize=(10,8))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title("Original img")
    

    ax[1].imshow(img_gray)
    ax[1].set_title("Gray img")
    
    plt.savefig(save_dir+"Original_and_Gray_img")
    plt.show()
    return img, img_gray
    
def binarization_and_closed(img_gray, save_dir, closed_iterations=2):   # binarization
    q75, q85 = np.percentile(img_gray.ravel(), [75, 85])
    ns, bins, *_ = plt.hist(img_gray.ravel())
    plt.vlines(q85, ymin=0, ymax=np.max(ns), colors="red")
    #plt.text(q85+5, np.mean(ns), s="q 85")
    plt.savefig(save_dir+"hist")
    
    ret, img_thresh = cv2.threshold(img_gray, q85, 255, cv2.THRESH_BINARY)
    
    
    # closed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel=kernel, iterations=closed_iterations)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(121)
    ax.imshow(img_thresh)
    ax.axis('off')
    ax.set_title("img binary")
    
    ax = fig.add_subplot(122)
    ax.imshow(img_closed)
    ax.set_title("img closed")
    plt.savefig(save_dir+"thresh_closed")
    
    return img_closed
    
### find contours ###

def find_contours(original_img, img_thresh, save_dir):
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_idx = np.argsort([len(arr) for arr in contours])
    img_tmp = original_img.copy()
    output = cv2.drawContours(img_tmp, contours[sorted_idx[-1]], -1, (0,0,255), 3) # a longest contours
    
    img_tmp = original_img.copy()
    cell_output = cv2.drawContours(img_tmp, contours[sorted_idx[-2]], -1, (0,255,0), 3)
    

    
    # plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(121)
    ax.imshow(output)
    ax.axis('off')
    ax.set_title("Contours")
    
    ax = fig.add_subplot(122)
    ax.imshow(cell_output)
    ax.set_title("Cell Body Contours")
    plt.savefig(save_dir+"Contours")
    
    return contours[sorted_idx[-1]], contours[sorted_idx[-2]]
    
    
def find_g(original_img, cell_contours, save_dir):
    cell_body = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
    cell_body = cv2.fillPoly(cell_body, cell_contours, 255)
    
    mu = cv2.moments(cell_body)
    x_g_cell, y_g_cell = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
    
    # plot
    img_tmp = original_img.copy()
    img_circle = cv2.circle(img_tmp, (x_g_cell,y_g_cell), radius=3, color=(255,0,0), thickness=-1)
    plt.figure(figsize=(8,8))
    plt.imshow(img_circle)
    plt.savefig(save_dir+"g")
    
    return x_g_cell, y_g_cell
### plot
def plot_thetas_rs(rs, labels, fig_title, save_dir, l=190):
    plt.figure(figsize=(20,8))
    plt.title(fig_title)
    for r, label in zip(rs, labels):
        plt.plot(r, label=label)
    plt.axvspan(0, l, alpha=0.2, color='m')
    plt.axvspan(l, 2*l, alpha=0.2, color='y')
    plt.axvspan(2*l, 3*l, alpha=0.2, color='r')
    plt.axvspan(3*l, 4*l, alpha=0.2, color='b')
    plt.legend()
    
    plt.savefig(save_dir+fig_title)
    
def plot_each_contours(original_img, axs, axs_cell, g, save_dir):
    img_tmp = original_img.copy()
    output = cv2.drawContours(img_tmp, axs+g, -1, (255,0,0),2)
    output = cv2.drawContours(img_tmp, axs_cell+g, -1, (0,255,0),2)
    img_circle = cv2.circle(img_tmp, (g[0], g[1]), radius=3, color=(255,0,0), thickness=-1)
    plt.figure(figsize=(8,8))
    plt.imshow(output)
    plt.axis("off")
    plt.savefig(save_dir+"reg_image")
    
def plot_peaks(rs, peaks, save_dir, fig_title="peaks"):
    plt.figure(figsize=(20,8))
    plt.plot(rs)
    plt.plot(peaks, rs[peaks], "x", markersize=20)
    plt.xlabel(f"detect {len(peaks)} blebs")
    plt.savefig(save_dir+fig_title)
    
    
def plot_peaks_img(original_img, peak_axs, g, bleb_area, area, save_dir, with_area=True, color="white", fig_title="area_of_each_bleb"):
    if with_area:
        img_tmp = original_img.copy()
        output = cv2.drawContours(img_tmp, peak_axs+g, -1, (255, 125 , 0), 2)
        plt.figure(figsize=(8,8))
        plt.imshow(output)
        plt.title(f"detect {len(peak_axs)} blebs")
        plt.axis('off')
        for (x,y),s in zip(peak_axs.reshape(-1, 2)+g, ["%.2f"%(a*100/area) for a in bleb_area]):
            plt.text(x+2, y+2, s, color=color)
    else:
        fig_title = "bleb_peaks"
        img_tmp = img.copy()
        output = cv2.drawContours(img_tmp, peak_axs+g, -1, (255, 125 , 0), 5)
        plt.figure(figsize=(8,8))
        plt.imshow(output)
        plt.title(f"detect {len(peak_axs)} blebs")
        plt.axis('off')
        
    plt.savefig(save_dir+fig_title)

def plot_series(r_consider_baseline, slided_rs, peaks, rel_mins, save_dir):
    plt.figure(figsize=(20,8))
    plt.plot(r_consider_baseline, label="rs", linestyle=":")
    plt.plot(slided_rs, label="slided_rs", linewidth=4)
    plt.plot(peaks, slided_rs[peaks], "x", markersize=20)
    plt.legend()
    plt.vlines(x=rel_mins, ymin=np.min(slided_rs), ymax=slided_rs[rel_mins], color="red", linestyle="--")
    plt.savefig(save_dir+"series")
    
    
## Shape characteristics

#def shape_characteristics()