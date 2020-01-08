# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:46:21 2018

@author: toppgabr
"""

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
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['font.size'] = 20# <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['figure.figsize'] = [10.,9.]
mpl.rcParams["legend.frameon"] = False
mpl.rcParams['text.usetex'] = True


RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 

def gauss(sigma, shift, x):
    return np.exp(-0.5*((x-shift)/sigma)**2)

ii = 4                                                                      # cell index
nn = 4*(ii**2+(ii+1)*ii+(ii+1)**2)                                             # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

file_BANDS = open('PROPAGATION/DATA/A_t.dat','r')
A_t = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/ASD_t.dat','r')
ASD = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/C_tL20.dat','r')
CL = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/C_tR20.dat','r')
CR = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/C_tL0.dat','r')
CL1 = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/C_tR0.dat','r')
CR1 = np.loadtxt(file_BANDS)
file_BANDS.close()

# =============================================================================
# file_BANDS = open('PROPAGATION/DATA/C_tL2.dat','r')
# CL2 = np.loadtxt(file_BANDS)
# file_BANDS.close()
# 
# file_BANDS = open('PROPAGATION/DATA/C_tR2.dat','r')
# CR2 = np.loadtxt(file_BANDS)
# file_BANDS.close()
# =============================================================================


file_BANDS = open('PROPAGATION/DATA/E_tL0.dat','r')
EL = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/E_tR0.dat','r')
ER = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/nv_tR0.dat','r')
NvR = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/nc_tR0.dat','r')
NcR = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/nv_tL0.dat','r')
NvL = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/nc_tL0.dat','r')
NcL = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/n_tR0.dat','r')
NtR = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS.close()

starttime = 0.0
endtime = 759.8
timesteps = 10000
T_PEAK = 500

reduce = 10
time = np.linspace(0,endtime,int(timesteps/reduce))*tc
h = (endtime-starttime)/int(timesteps/reduce)


sigma = 90.

# =============================================================================
# C0x_AV= np.zeros(int(timesteps/reduce))
# C0y_AV= np.zeros(int(timesteps/reduce))
# 
# Cx_AV= np.zeros(int(timesteps/reduce))
# Cy_AV= np.zeros(int(timesteps/reduce))
# for t in range(int(timesteps/reduce)):
#     for ts in range(int(timesteps/reduce)):
#         Cx_AV[t] += 1./np.sqrt(2.*np.pi*sigma**2)*0.25*((CR[ts-1,0]-CL[ts-1,0])*np.exp(-((t-ts+1)*h)**2/(2.*sigma**2)) + (CR[ts,0]-CL[ts,0])*np.exp(-((t-ts)*h)**2/(2.*sigma**2)))          
#         Cy_AV[t] += 1./np.sqrt(2.*np.pi*sigma**2)*0.25*((CR[ts-1,1]-CL[ts-1,1])*np.exp(-((t-ts+1)*h)**2./(2.*sigma**2)) + (CR[ts,1]-CL[ts,1])*np.exp(-((t-ts)*h)**2/(2.*sigma**2))) 
# 
# =============================================================================
fig1 = plt.figure(2)
gs1 = gridspec.GridSpec(4, 1)
#fig1.suptitle(r'$\mathrm{\Theta = '+str(np.round(THETA*360/(2*np.pi),2))+'^\circ}$, $\mathrm{N_{atom}='+str(nn)+'}$, $\mathrm{\omega}=0.2eV$, $\mathrm{A_{max}}=0.05 a_0^{-1}$, $\mathrm{E_{max}}=0.0005 MV/cm$, $\mathrm{E_{SD}}=0.0004 MV/m$',fontsize=15)

ax1 = fig1.add_subplot(gs1[0,0])
ax1.plot(time, ASD[:,0]/ASD[-1,0]*5, linewidth=2.0, color='k', label=r"$\mathrm{E_{SD}(t)}$ $\mathrm{\times}$ $\mathrm{500}$")
#ax1.plot(time, A_t[:,0], linewidth=2.0, color='k', label="$A_{x}(t)$")
#ax1.plot(time, A_t[:,1], linewidth=2.0, color='k', label="$A_{y}(t)$")
#ax1.plot(time, ASD[:,1]-ASD[0,1], linewidth=2.0, color=BLUE, label="$A_{SD}(t)$")
#ax1.set_xlabel("$\mathrm{time}$  ($\mathrm{fs}$)")
ax1.fill(time,5*pow(np.sin(np.pi*time/T_PEAK),2.)*(1.-np.heaviside(time-T_PEAK, 0)), color="gainsboro", facecolor='gainsboro')
ax1.plot(time,5*pow(np.sin(np.pi*time/T_PEAK),2.)*(1.-np.heaviside(time-T_PEAK, 0)), 'k:', linewidth=1.5, alpha=0.5,label=r"$\mathrm{Circular}$ $\mathrm{pump}$")
ax1.set_ylabel("$\mathrm{E(t)}$ ($\mathrm{MV/m}$)")
ax1.text(0.015, 0.95, r'(a)', fontsize=18, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)
ax1.get_yaxis().set_label_coords(-0.14,0.5)
ax1.set_xticklabels([])
plt.legend(loc="center right")

# =============================================================================
# ax2 = fig1.add_subplot(gs1[5,0])
# ax2.plot(time, EL-EL[0], linewidth=2.0, linestyle="-", color=BLUE, label="$E_L(t)-E_L(0)$")
# ax2.plot(time, ER-ER[0], linewidth=2.0, linestyle="-", color=RED, label="$E_R(t)-E_R(0)$")
# #ax2.plot(time, E0L-E0L[0], linewidth=2.0, linestyle="--", color=GREEN, label="$E0_L(t)-E0_L(0)$")
# #ax2.fill(time,abs(np.amax(E0L)-E0L[0])*gauss(100.*tc, 500*tc, time), color="gainsboro", facecolor='gainsboro',label=r"$\mathrm{pump}$",alpha=0.5)
# ax2.set_ylabel("$\mathrm{\Delta E}$ ($\mathrm{eV}$)")
# ax2.set_ylim(-1e-4, 1e-4)
# plt.legend(loc="upper right")
# =============================================================================

# =============================================================================
# ax3 = fig1.add_subplot(gs1[2,0])
# ax3.set_title("No $\mathrm{SD-Field}$ $\mathrm{only}$ ")
# #ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
# #ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
# #ax2.plot(time, C0L[:,0]-C0R[:,0], linewidth=2.0, color=BLUE, label=r"$J_x(L-R)$")
# ax3.plot(time, C0x_AV, linewidth=2.0, color=BLUE, label=r"$J0_x(L-R)_{AV}$")
# ax3.plot(time, C0y_AV, linewidth=2.0, color=RED, label=r"$J0_y(L-R)_{AV}$")
# #ax2.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
# ax3.set_ylabel("$\mathrm{J}$ ($\mathrm{arb. units}$)")
# plt.legend(loc="upper right")
# =============================================================================


ax3 = fig1.add_subplot(gs1[1,0])
#ax3.set_title("Left circular")
#ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
#ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
ax3.plot(time, CL[:,0], 'k-', linewidth=2.0, label=r"$\mathrm{Left}$ $\mathrm{polarized}$")
ax3.plot(time, CR[:,0], color=RED, linestyle='-', linewidth=2.0, label=r"$\mathrm{Right}$ $\mathrm{polarized}$")
ax3.plot(time, [0]*time,  'k:', linewidth=0.5)
#ax3.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
ax3.set_ylabel("$\mathrm{J_x}$ ($\mathrm{10^{-10} Cm/s}$)")
ax3.text(0.015, 0.95, r'(b)', fontsize=18, horizontalalignment='left', verticalalignment='top', transform=ax3.transAxes)
ax3.get_yaxis().set_label_coords(-0.14,0.5)
ax3.set_xticklabels([])
plt.legend(loc="upper right")

ax4 = fig1.add_subplot(gs1[2,0])
#ax4.set_title("Right circular")
#ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
#ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
ax4.plot(time, CL[:,1], 'k-', linewidth=2.0, label=r"$\mathrm{Left}$ $\mathrm{polarized}$")
ax4.plot(time, CR[:,1], color=RED, linestyle='--', linewidth=2.0, label=r"$\mathrm{Right}$ $\mathrm{polarized}$")
ax4.plot(time, [0]*time,  'k:', linewidth=0.5)
ax4.text(0.015, 0.95, r'(c)', fontsize=20, horizontalalignment='left', verticalalignment='top', transform=ax4.transAxes)
#ax4.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
ax4.set_ylabel("$\mathrm{J_y}$ ($\mathrm{10^{-10} Cm/s}$)")
ax4.get_yaxis().set_label_coords(-0.14,0.5)
ax4.set_xticklabels([])
plt.legend(loc="upper right")

ax5 = fig1.add_subplot(gs1[3,0])
#ax5.set_title("$\mathrm{Hall}$ $\mathrm{current}$ $\mathrm{av.}$")
#ax5.plot(time, 0.5*(CL[:,0]-CL[:,1]-(CR[:,0]-CR[:,1])), linewidth=2.0, color=BLUE, label="$J_x(L-R)$")
ax5.plot(time, 0.5*(CR[:,1]-CL[:,1]), linewidth=2.0, color=RED, label="$\mathrm{E_{SD}(t)}$")
#ax5.plot(time, 0.5*(CR2[:,1]-CL2[:,1]), linewidth=2.0, color=RED, label="$\mathrm{E_{SD}(t)} (1st order)$")
ax5.plot(time, 0.5*(CR1[:,1]-CL1[:,1]), linewidth=2.0, color='k', label="$\mathrm{-E_{SD}(t)}$")
#ax5.plot(time, 0.5*((CR[:,0]-CR[:,1])-(CR[:,2]-CR[:,3]) - (CL[:,0]-CL[:,1])+(CL[:,2]-CL[:,3])), linewidth=2.0, color=RED, label="$J_y(L-R)$")

#ax5.plot(time, 0.5*(CL[:,1]-CR[:,1]), linewidth=2.0, color=RED, label="$J_y(L-R)$")
#ax5.plot(time, Cx_AV, linewidth=2.0, color=BLUE, linestyle='--', label="$J_x(L-R)_{AV}$")
#ax5.plot(time, C0y_AV, linewidth=2.0, color=BLUE, linestyle='-', label=r"$E_{SD}=0$")
#ax5.plot(time, Cy_AV, linewidth=2.0, color=RED, linestyle='-',label=r"$<J^L_y-J^R_y>$")
#ax5.fill(time,abs(np.amax(Cy_AV)-Cy_AV[0])*gauss(100.*tc, 500*tc, time), color="gainsboro", facecolor='gainsboro',alpha=0.5)
#ax1.plot(time, ASD[:,1]-ASD[0,1], linewidth=2.0, color=BLUE, label="$A_{SD}(t)$")
ax5.set_ylabel("$\mathrm{J_{Hall}}$ ($\mathrm{10^{-10} Cm/s}$)")
ax5.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
ax5.get_yaxis().set_label_coords(-0.14,0.5)
ax5.plot(time, [0]*time,  'k:', linewidth=0.5)
ax5.text(0.015, 0.95, r'(d)', fontsize=18, horizontalalignment='left', verticalalignment='top', transform=ax5.transAxes)
plt.legend(loc="upper right")

# =============================================================================
# ax6 = fig1.add_subplot(gs1[4,0])
# ax6.set_title("Linear polarized")
# #ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
# #ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
# #ax6.plot(time, NvR+NcR, linewidth=2.0, color=BLUE, label=r"$N_{tot}$")
# ax6.plot(time, NvR-NvR[0], linewidth=2.0, color=BLUE, label=r"$N_{vR}$")
# ax6.plot(time, NcR-NcR[0], linewidth=2.0, color=RED, label=r"$N_{cR}$")
# #ax6.plot(time, NcR-NcR[0], linewidth=2.0, color=BLUE, linestyle="--", label=r"$N_{cR}$")
# #ax6.plot(time, NvL, linewidth=2.0, color=RED, label=r"$N_{vL}$")
# #ax6.plot(time, NcL, linewidth=2.0, color=RED, linestyle="--", label=r"$N_{cL}$")
# #ax3.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
# ax6.set_ylabel("$\mathrm{\Delta N_c}$")
# plt.legend(loc="upper right")
# =============================================================================


plt.subplots_adjust(top=0.99, bottom=0.10, left=0.15, right=0.98, hspace=0.20, wspace=0.20)
#plt.tight_layout()
plt.show()

