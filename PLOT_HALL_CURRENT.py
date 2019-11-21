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
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['font.size'] = 14  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['figure.figsize'] = [12.,12]


RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 

def gauss(sigma, shift, x):
    return np.exp(-0.5*((x-shift)/sigma)**2)

ii = 4                                                                        # cell index
nn = 4*(ii**2+(ii+1)*ii+(ii+1)**2)                                             # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

file_BANDS = open('PROPAGATION/DATA/200/ASD_t.dat','r')
ASD = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/200/C_tL0.dat','r')
CL = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/200/C_tR0.dat','r')
CR = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/200/E_tL0.dat','r')
EL = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('PROPAGATION/DATA/200/E_tR0.dat','r')
ER = np.loadtxt(file_BANDS)
file_BANDS.close()

starttime = 0.0
endtime = 1000.
timesteps = 100000


reduce = 20
time = np.linspace(0,1000,int(timesteps/reduce))*tc
h = (endtime-starttime)/int(timesteps/reduce)
ASD = ASD[::reduce]


sigma = 100.
# =============================================================================
#C0x_AV= np.zeros(int(timesteps/reduce))
#C0y_AV= np.zeros(int(timesteps/reduce))
# =============================================================================
#Cx_AV= np.zeros(int(timesteps/reduce))
#Cy_AV= np.zeros(int(timesteps/reduce))
#for t in range(int(timesteps/reduce)):
#    for ts in range(int(timesteps/reduce)):
#        Cx_AV[t] += 1./np.sqrt(2.*np.pi*sigma**2)*0.5*((CL[ts-1,0]-CR[ts-1,0])*np.exp(-((t-ts+1)*h)**2/(2.*sigma**2)) + (CL[ts,0]-CR[ts,0])*np.exp(-((t-ts)*h)**2/(2.*sigma**2)))*h          
#        Cy_AV[t] += 1./np.sqrt(2.*np.pi*sigma**2)*0.5*((CL[ts-1,1]-CR[ts-1,1])*np.exp(-((t-ts+1)*h)**2./(2.*sigma**2)) + (CL[ts,1]-CR[ts,1])*np.exp(-((t-ts)*h)**2/(2.*sigma**2)))*h 

fig1 = plt.figure(2)
gs1 = gridspec.GridSpec(5, 1)
fig1.suptitle(r'$\mathrm{\Theta = '+str(np.round(THETA*360/(2*np.pi),2))+'^\circ}$, $\mathrm{N_{atom}='+str(nn)+'}$, $\mathrm{\omega}=0.2eV$, $\mathrm{A_{max}}=0.05 a_0^{-1}$, $\mathrm{E_{max}}=0.0005 MV/cm$, $\mathrm{E_{SD}}=0.0004 MV/m$',fontsize=15)

ax1 = fig1.add_subplot(gs1[0,0])
ax1.plot(time, ASD[:,0]-ASD[0,0], linewidth=2.0, color='k', label="$E_{SD}(t)$")
#ax1.plot(time, ASD[:,1]-ASD[0,1], linewidth=2.0, color=BLUE, label="$A_{SD}(t)$")
#ax1.set_xlabel("$\mathrm{time}$  ($\mathrm{fs}$)")
ax1.fill(time,abs(np.amax(ASD[:,0])-ASD[0,0])*gauss(100.*tc, 500*tc, time), color="gainsboro", facecolor='gainsboro',label=r"$\mathrm{pump}$",alpha=0.5)
ax1.set_ylabel("$\mathrm{SD-Field}$ ($\mathrm{arb. units}$)")
plt.legend(loc="upper right")

ax2 = fig1.add_subplot(gs1[4,0])
ax2.plot(time, EL-EL[0], linewidth=2.0, linestyle="-", color=BLUE, label="$E_L(t)-E_L(0)$")
ax2.plot(time, ER-ER[0], linewidth=2.0, linestyle="-", color=RED, label="$E_R(t)-E_R(0)$")
#ax2.plot(time, E0L-E0L[0], linewidth=2.0, linestyle="--", color=GREEN, label="$E0_L(t)-E0_L(0)$")
#ax2.fill(time,abs(np.amax(E0L)-E0L[0])*gauss(100.*tc, 500*tc, time), color="gainsboro", facecolor='gainsboro',label=r"$\mathrm{pump}$",alpha=0.5)
ax2.set_ylabel("$\mathrm{\Delta E}$ ($\mathrm{eV}$)")
plt.legend(loc="upper right")

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

# =============================================================================
# ax3 = fig1.add_subplot(gs1[2,0])
# ax3.set_title("Linear polarized")
# #ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
# #ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
# ax3.plot(time, C0L[:,0], linewidth=2.0, color=BLUE, label=r"$J_x$")
# ax3.plot(time, C0L[:,1], linewidth=2.0, color=RED, label=r"$J_y$")
# #ax3.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
# ax3.set_ylabel("$\mathrm{J}$ ($\mathrm{arb. units}$)")
# plt.legend(loc="upper right")
# =============================================================================

ax3 = fig1.add_subplot(gs1[1,0])
#ax3.set_title("Left circular")
#ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
#ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
ax3.plot(time, CL[:,0], linewidth=2.0, color=BLUE, label=r"$J^L_x$")
ax3.plot(time, CR[:,0], linewidth=2.0, color=RED, label=r"$J^R_x$")
ax3.plot(time, [0]*time,  'k--', linewidth=0.5)
#ax3.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
ax3.set_ylabel("$\mathrm{J}$ ($\mathrm{arb. units}$)")
plt.legend(loc="upper right")

ax4 = fig1.add_subplot(gs1[2,0])
#ax4.set_title("Right circular")
#ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
#ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
ax4.plot(time, CL[:,1], linewidth=2.0, color=BLUE, label=r"$J^L_x$")
ax4.plot(time, CR[:,1], linewidth=2.0, color=RED, label=r"$J^R_y$")
ax4.plot(time, [0]*time,  'k--', linewidth=0.5)
#ax4.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
ax4.set_ylabel("$\mathrm{J}$ ($\mathrm{arb. units}$)")
plt.legend(loc="upper right")

ax5 = fig1.add_subplot(gs1[3,0])
#ax5.set_title("$\mathrm{Hall}$ $\mathrm{current}$ $\mathrm{av.}$")
ax5.plot(time, CL[:,0]+CR[:,0], linewidth=2.0, color=BLUE, label="$J_x(L+R)$")
#ax5.plot(time, CL[:,1]+CR[:,1], linewidth=2.0, color=RED, label="$J_y(L+R)$")
#ax5.plot(time, Cx_AV, linewidth=2.0, color=BLUE, linestyle='--', label="$J_x(L-R)_{AV}$")
#ax5.plot(time, C0y_AV, linewidth=2.0, color=BLUE, linestyle='-', label=r"$E_{SD}=0$")
#ax5.plot(time, Cy_AV, linewidth=2.0, color=RED, linestyle='-',label=r"$<J^L_y-J^R_y>$")
#ax5.fill(time,abs(np.amax(Cy_AV)-Cy_AV[0])*gauss(100.*tc, 500*tc, time), color="gainsboro", facecolor='gainsboro',alpha=0.5)
#ax1.plot(time, ASD[:,1]-ASD[0,1], linewidth=2.0, color=BLUE, label="$A_{SD}(t)$")
ax5.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
ax5.set_ylabel("$\mathrm{J_{L}}+\mathrm{J_{R}}$ ($\mathrm{arb. units}$)")
ax5.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
plt.legend(loc="upper right")

# =============================================================================
# ax6 = fig1.add_subplot(gs1[4,0])
# #ax5.set_title("$\mathrm{Hall}$ $\mathrm{current}$ $\mathrm{av.}$")
# ax6.plot(time, CL[:,0]-CR[:,0], linewidth=2.0, color=BLUE, label="$J_x(L-R)$")
# ax6.plot(time, CL[:,1]-CR[:,1], linewidth=2.0, color=RED, label="$J_y(L-R)$")
# #ax6.plot(time, Cx_AV, linewidth=2.0, color=BLUE, linestyle='--', label="$J_x(L-R)_{AV}$")
# #ax5.plot(time, C0y_AV, linewidth=2.0, color=BLUE, linestyle='-', label=r"$E_{SD}=0$")
# #ax6.plot(time, Cy_AV, linewidth=2.0, color=RED, linestyle='-',label=r"$<J^L_y-J^R_y>$")
# #ax5.fill(time,abs(np.amax(Cy_AV)-Cy_AV[0])*gauss(100.*tc, 500*tc, time), color="gainsboro", facecolor='gainsboro',alpha=0.5)
# #ax1.plot(time, ASD[:,1]-ASD[0,1], linewidth=2.0, color=BLUE, label="$A_{SD}(t)$")
# ax6.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
# ax6.set_ylabel("$\mathrm{J_{HALL}}$ ($\mathrm{arb. units}$)")
# ax6.set_xlabel("$\mathrm{time}$ ($\mathrm{fs}$)")
# plt.legend(loc="upper right")
# =============================================================================

plt.tight_layout()
plt.legend(loc="upper right")
plt.subplots_adjust(top=0.92)
plt.show()

