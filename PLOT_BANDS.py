# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:00:25 2018

@author: toppgabr
"""

import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 20  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['figure.figsize'] = [8.,5]
mpl.rcParams['text.usetex'] = True

#m = 1

RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 

window = 2.0
omega = 0.8

ii = 4                                                                 # cell index
nn = int(4*(ii**2+(ii+1)*ii+(ii+1)**2))                                             # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

num_GK = 43                                                                    # number of k-point per hgh symmetry line
num_KM = 42   

file_BANDS = open('mu.dat','r')
mu = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('bands.dat','r')
MAT_BANDS = np.loadtxt(file_BANDS)-mu
file_BANDS.close()

for k in range(np.size(MAT_BANDS[:,0])):
    for i in range(np.size(MAT_BANDS[0,:])):    
        if(MAT_BANDS[k,i] > 0.3 or MAT_BANDS[k,i] < -0.3):
           MAT_BANDS[k,i] = 'nan'

MAT_BANJDS_CUT = MAT_BANDS        
        
file_BANDS = open('bands.dat','r')
MAT_BANDS = np.loadtxt(file_BANDS)-mu
file_BANDS.close()

print(np.shape(MAT_BANDS))

fig1 = plt.figure(1)
gs1 = gridspec.GridSpec(1, 1)
ax11 = fig1.add_subplot(gs1[0,0])

ax11.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
#ax12.set_xticks([0 , num_GK, num_GK+num_KM])
#ax12.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{K1}$' , r'$\mathrm{M}$'])
ax11.set_xticks([0 , num_GK, num_GK+num_KM/2, num_GK+num_KM, 2*num_GK+num_KM])
ax11.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{K1}$' , r'$\mathrm{M}$', r'$\mathrm{K2}$', '$\mathrm{\Gamma}$'])
#ax12.plot([0]*np.size(MAT_BANDS[:,0]), 'k--', linewidth=1.0)
ax11.plot(MAT_BANDS[:,:], 'k', linewidth=1.0, alpha=1.0)
ax11.plot(MAT_BANJDS_CUT[:,:], color=RED, linewidth=3.0)
ax11.hlines(y=0.3, xmin=0, xmax = np.size(MAT_BANDS[:,0]), color='k', linestyle=":", linewidth=1.0)
ax11.hlines(y=-0.3, xmin=0, xmax = np.size(MAT_BANDS[:,0]), color='k', linestyle=":", linewidth=1.0)
ax11.vlines(x=32.7, ymin=-2, ymax = 2, color='k', linestyle=":", linewidth=1.0)
ax11.vlines(x=53.3, ymin=-2, ymax = 2, color='k', linestyle=":", linewidth=1.0)
ax11.vlines(x=75, ymin=-2, ymax = 2, color='k', linestyle=":", linewidth=1.0)
ax11.vlines(x=95, ymin=-2, ymax = 2, color='k', linestyle=":", linewidth=1.0)
ax11.text(0.055, 0.47, r'$\mathrm{\Delta_E} = $ '+str(0.6)+' $\mathrm{eV}$', color='k', fontsize=15, horizontalalignment='left', verticalalignment='bottom', transform=ax11.transAxes)
ax11.annotate("", xy=(0.0, -0.3), xytext=(0.0, +0.3), arrowprops=dict(arrowstyle="<->", color='k'))
ax11.set_ylim(-omega*window,+omega*window) 

plt.tight_layout()

plt.show()
