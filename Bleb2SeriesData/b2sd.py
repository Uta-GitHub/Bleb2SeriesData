# -*- coding: utf-8 -*-
"""
Created on Sat July 30 15:51:44 2022

@author: Yuta Takahashi
"""


### transform Bleb images to Series Datas ###
import sys
sys.path.append("../Bleb2SeriesData/")

import numpy as np
import scipy.signal as signal
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from scipy import signal, interpolate
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from Bleb2SeriesData.utils import plot_thetas_rs

import pandas as pd


def xy2theta(contours, g):
    """
    xy座標からΘ座標の算出
    
    returns
        sorted_contours : 各象限のソートされた輪郭点
        thetas : 各象限のΘ
        ls : 各象限のcontour数
        
    """
    new_c = contours.reshape(-1, 2) - g.reshape(2,)
    
    
    
    # 象限ごとのlistを用意
    theta_1 = [] # 第一象限
    theta_2 = []
    theta_3 = []
    theta_4 = []
    idx_1 = []
    idx_2 = []
    idx_3 = []
    idx_4 = []
    # 各象限ごとにthetaを格納（arctanで求める）ax[1]==0, ax[0]==0
    for i, ax in enumerate(new_c):
    
        if ax[0]>0 and ax[1]>0:
            theta_1.append(np.arctan(ax[1]/ax[0])) # return : 0 ~ pi/2
            idx_1.append(i)
        elif ax[0]<0 and ax[1]>0:
            theta_2.append(np.arctan(ax[1]/ax[0])) # return : -pi/2 ~ 0??
            idx_2.append(i)
        elif ax[0]<0 and ax[1]<0:
            theta_3.append(np.arctan(ax[1]/ax[0])) # return: 0~pi/2 ??
            idx_3.append(i)
        elif ax[0]>0 and ax[1]<0:
            theta_4.append(np.arctan(ax[1]/ax[0])) # return : -pi/2~ 0
            idx_4.append(i)
    # df作成
    df_1 = pd.DataFrame([theta_1, idx_1]).T.sort_values(by=0)
    df_2 = pd.DataFrame([theta_2, idx_2]).T.sort_values(by=0)
    df_3 = pd.DataFrame([theta_3, idx_3]).T.sort_values(by=0)
    df_4 = pd.DataFrame([theta_4, idx_4]).T.sort_values(by=0)
    
    ls = [] # 各象限のcontour数
    sorted_contours = []
    thetas = []
    
    for df in [df_1, df_2, df_3, df_4]:
        idx = df.iloc[:,1].values.astype(int)
        c, theta, l = contours[idx], df.iloc[:,0].values , len(df)
        
        sorted_contours.append(c)
        thetas.append(theta)
        ls.append(l)
    
    return sorted_contours, thetas, ls
    

def cal_rs(contours, g):
    """重心gからの距離Rを算出"""
    rs = []
    for c in contours:
        c = c.reshape(2)
        r = np.sqrt((c[0]-g[0])**2 + (c[1]-g[1])**2)
        rs.append(r)
    return rs


def contour_regression(contours, cell_contours, g, save_dir, degree=40, n_conv=25):
    sorted_c, thetas, ls = xy2theta(contours, g)
    sorted_cell_c, thetas_cell, ls_cell = xy2theta(cell_contours, g)
    
    all_r = []
    each_r = []
    each_theta = []
    
    for i in range(4):
        rs = cal_rs(sorted_c[i], g)
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        theta_poly = poly_features.fit_transform(thetas[i].reshape(-1,1))
        reg = LinearRegression().fit(theta_poly, rs)
        if i%2!=0: new_theta = np.linspace(-np.pi/2, 0, 200) # indexが1,3なので,2,4象限であることに注意！
        elif i%2==0: new_theta = np.linspace(0, np.pi/2, 200) # indexが0,2なので1,3象限であることに注意！！
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        new_theta_poly = poly_features.fit_transform(new_theta.reshape(-1,1))
        reg_r = reg.predict(new_theta_poly)[5:-5]
        all_r.extend(reg_r.tolist())
        each_r.append(reg_r.tolist())
        each_theta.append(new_theta[5:-5])
    
    each_r_cell = []
    each_theta_cell = []
    all_cell_r = []
    for i in range(4):
        rs_cell = cal_rs(sorted_cell_c[i], g)
        poly_features_cell = PolynomialFeatures(degree=5, include_bias=False)
        theta_poly_cell = poly_features_cell.fit_transform(thetas_cell[i].reshape(-1,1))
        reg_cell = LinearRegression().fit(theta_poly_cell, rs_cell)
        if i%2!=0: new_theta_cell = np.linspace(-np.pi/2, 0, 200) # indexが1,3なので,2,4象限であることに注意！
        elif i%2==0: new_theta_cell = np.linspace(0, np.pi/2, 200) # indexが0,2なので1,3象限であることに注意！！
        poly_features_cell = PolynomialFeatures(degree=5, include_bias=False)
        new_theta_poly_cell = poly_features_cell.fit_transform(new_theta_cell.reshape(-1,1))
        reg_r = reg_cell.predict(new_theta_poly_cell)[5:-5]
        all_cell_r.extend(reg_r.tolist())
    
        
    
        
        each_r_cell.append(reg_r.tolist()) #####
        each_theta_cell.append(new_theta_cell[5:-5])####
        
    
    # 移動平均を出す
    b = np.ones(n_conv)/n_conv

    all_r_padded = np.concatenate((all_r[-20:], all_r, all_r[:20]))
    all_r_mean = np.convolve(all_r_padded, b, mode="same")[20:-20]
    
    all_cell_r_padded = np.concatenate((all_cell_r[-20:], all_cell_r, all_cell_r[:20]))
    all_cell_r_mean = np.convolve(all_cell_r_padded, b, mode="same")[20:-20]
    
    # plot
    plot_thetas_rs([all_r, all_r_mean], ["reg", "mean"], fig_title="bleb_contours_series", save_dir=save_dir)
    plot_thetas_rs([all_r_mean, all_cell_r_mean], ["mean", "cell body mean"], fig_title="bleb_cell_contours", save_dir=save_dir)
    
    return each_r, each_theta, all_r,all_r_mean, each_r_cell, each_theta_cell, all_cell_r, all_cell_r_mean
    
    


def contour_interpolation(contours, cell_contours, g, save_dir, n_conv=25, mean_iter=2):
    sorted_c, thetas, ls = xy2theta(contours, g)
    sorted_cell_c, thetas_cell, ls_cell = xy2theta(cell_contours, g)
    
    all_r = []
    each_r = []
    each_theta = []
    ind_s = 0
    for i in range(4):
        rs = cal_rs(sorted_c[i], g)
        theta = thetas[i]
        
        #print(int(ls[i]), np.array(thetas[i]).shape, np.array(rs).shape)
        #print(np.unique(np.array(theta)).shape, len(theta))
        
        
        if len(np.unique(theta)) != len(theta):# 重複を削除する
            u, ind = np.unique(theta, return_index=True)
            ind_sorted = np.sort(ind)
            theta = theta[ind_sorted]
            rs = np.array(rs)[ind_sorted]
        
        inp = interpolate.interp1d(theta, rs, kind="quadratic")
        ind_s = ls[i]
        if i%2!=0: new_theta = np.linspace(-np.pi/2, 0, 200)[5:-5] # indexが1,3なので,2,4象限であることに注意！
        elif i%2==0: new_theta = np.linspace(0, np.pi/2, 200)[5:-5] # indexが0,2なので1,3象限であることに注意！！
        
        rs_inp = inp(new_theta)
        all_r.extend(rs_inp.tolist())
        each_r.append(rs_inp.tolist())
        each_theta.append(new_theta)
    
    each_r_cell = []
    each_theta_cell = []
    all_cell_r = []
    ind_s = 0
    
    for i in range(4):
        rs_cell = cal_rs(sorted_cell_c[i], g)
        theta_cell = thetas_cell[i]

        if len(np.unique(theta_cell)) != len(theta_cell):# 重複を削除する
            u, ind = np.unique(theta_cell, return_index=True)
            ind_sorted = np.sort(ind)
            theta_cell = theta_cell[ind_sorted]
            
            rs_cell = np.array(rs_cell)[ind_sorted]
        inp = interpolate.interp1d(theta_cell, rs_cell, kind="quadratic")
        ind_s = ls_cell[i]
        
        if i%2!=0: new_theta_cell = np.linspace(-np.pi/2, 0, 200)[5:-5] # indexが1,3なので,2,4象限であることに注意！
        elif i%2==0: new_theta_cell = np.linspace(0, np.pi/2, 200)[5:-5] # indexが0,2なので1,3象限であることに注意！！

        rs_inp = inp(new_theta_cell)
        
        all_cell_r.extend(rs_inp.tolist())
        each_r_cell.append(rs_inp.tolist()) #####
        each_theta_cell.append(new_theta_cell)####
        
    
    # 移動平均を出す
    b = np.ones(n_conv)/n_conv
    tmp = all_r
    
    for _ in range(mean_iter):
        all_r_padded = np.concatenate((tmp[-20:], tmp, tmp[:20]))
        all_r_mean = np.convolve(all_r_padded, b, mode="same")[20:-20]
    
    #all_cell_r_padded = np.concatenate((all_cell_r[-20:], all_cell_r, all_cell_r[:20]))
    #all_cell_r_mean = np.convolve(all_cell_r_padded, b, mode="same")[20:-20]
    cell_tmp = all_cell_r
    for _ in range(mean_iter):
        all_cell_r_padded = np.concatenate((cell_tmp[-20:], cell_tmp, cell_tmp[:20]))
        all_cell_r_mean = np.convolve(all_cell_r_padded, b, mode="same")[20:-20]
        cell_tmp = all_cell_r_mean
    
    
    # plot
    plot_thetas_rs([all_r, all_r_mean], ["reg", "mean"], fig_title="bleb_contours_series", save_dir=save_dir)
    plot_thetas_rs([all_r_mean, all_cell_r_mean], ["mean", "cell body mean"], fig_title="bleb_cell_contours", save_dir=save_dir)
    
    return each_r, each_theta, all_r,all_r_mean, each_r_cell, each_theta_cell, all_cell_r, all_cell_r_mean
    
    
## 画像に戻す

def old_theta2xy(each_r, each_theta):
    """r,theta(極座標）からxy座標に変換"""
    axs = []
    for i in range(4):#象限ごとに戻す 
        r = each_r[i]
        theta = each_theta[i]
        if i==0 or i==3: # 1,4象限 
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            
        elif i==1 or i==2: # 2,3象限 
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            
            x = -x
            y = -y
        
        ax = np.c_[x,y]
        
        axs.extend(ax)
    axs = np.array(axs)[:, np.newaxis, :] # contoursのshapeに合わせる(N, 1, 2)
    axs = axs.astype(np.int32)
    return  axs
    
def theta2xy(all_r_mean, each_theta):
    """r,theta(極座標）からxy座標に変換"""
    #print(len(all_r_mean))
    l = int(len(all_r_mean)/4)
    
    
    axs = []
    for i in range(4):#象限ごとに戻す 
        each_r_mean = all_r_mean[l*i : l*(i+1)]
        r = each_r_mean
        theta = each_theta[i]
        #print("len",len(r), len(theta))
        if i==0 or i==3: # 1,4象限 
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            
        elif i==1 or i==2: # 2,3象限 
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            
            x = -x
            y = -y
        
        ax = np.c_[x,y]
        
        axs.extend(ax)
    axs = np.array(axs)[:, np.newaxis, :] # contoursのshapeに合わせる(N, 1, 2)
    axs = axs.astype(np.int32)
    return  axs