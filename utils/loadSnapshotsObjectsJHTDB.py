import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable 
from matplotlib import colors


#----------------------------------Visualization---------------------------------#

def plotContour_fast(x, y, numberx, numbery, path, time, snapshots, cmaps, pltT, levels, minz, maxz, title, ob):
    norm1 = colors.Normalize(vmin=minz, vmax=maxz, clip = True)
    # norm = colors.Normalize(minz, maxz)
    if isinstance(snapshots,list):
        series = enumerate(snapshots[::pltT], start=1)
    else:
        S_list =[]
        for m in range(np.shape(snapshots)[1]):
            S_list.append(snapshots[:,m])
        snapshots = S_list
        series = enumerate(snapshots[::pltT], start=1)
    for i, snapshot in series:
        fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
        ax.set_title('\n {} field at {}s\n'.format(title, time[::pltT][i-1]),size=60) 
        ss = np.reshape(snapshots[i], (numbery, numberx), order = 'C')
        plt.contourf(x, y, ss, 100, cmap = cmaps, norm=norm1)
        # cset1 = ax.contourf(x, y, ss, 100, cmap = cmaps, norm=norm1)
        mappable = ScalarMappable(norm=norm1, cmap=cmaps)
        cbar = fig.colorbar(mappable=mappable, ax=ax, spacing='uniform')
        # cbar = plt.colorbar(cset1, spacing='uniform')
        # cbar = plt.colorbar()
        # plt.clim(minz, maxz)
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_xticks(np.linspace(0, 2*np.pi, 3),[r'$0$', r'$\pi$', r'$2\pi$'],size=40)
        ax.set_yticks(np.linspace(0, 2*np.pi, 3),[r'$0$', r'$\pi$', r'$2\pi$'],size=40)
        ax.set_aspect("equal")
        cbar.set_ticks(levels)
        cbar.set_label('{}'.format(ob), size=60)
        cbar.ax.tick_params(labelsize=40)
        
        plt.show()      
        fig.savefig(path+r'{} contour_{}.jpg'.format(title, time[::pltT][i-1]))
        plt.close('all')
    return

def plotContourErr_fast(x, y, numberx, numbery, path, time, snapshots, cmaps, pltT, levels, minz, maxz, ob):
    for i, snapshot in enumerate(snapshots[::pltT], start=1):
        ss = np.reshape(snapshots[i], (numbery, numberx), order = 'C')
        fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
        ax.set_title('\n {} contour of vortex shedding at {}s\n'.format(ob, time[::pltT][i-1]),size=32) 
        plt.contourf(x,y,ss,100, cmap = cmaps, level=levels)
        cset1 = ax.contourf(x,y,ss,100, cmap = cmaps, level=levels, extend="both")
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_xticks(np.linspace(0, 2*np.pi, 3),[r'$0$', r'$\pi$', r'$2\pi$'],size=40)
        ax.set_yticks(np.linspace(0, 2*np.pi, 3),[r'$0$', r'$\pi$', r'$2\pi$'],size=40)
        cbar = plt.colorbar(cset1)
        cbar.set_label('{}'.format(ob), size=30)
        cbar.set_ticks(levels)
        # cbar.set_ticks(np.linspace(minz, maxz, 10))
        # cbar.ax.yaxis.set_major_locator(plt.MultipleLocator(tick))
        # cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        plt.show()      
        fig.savefig(path+r'{} contour_{}.jpg'.format(ob, time[::pltT][i-1]))
        plt.close('all')
    return

def plot_temporal_rmse_fast(x, y, numberx, numbery, cmaps, rmse, path, method, ob):
    rmse = np.reshape(rmse, (numbery, numberx), order = 'F')
    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
    ax.set_title(('\n Temporal-averaged {} of '.format(ob) + method + ' solutions \n'),size=60) 
    plt.contourf(x, y, rmse, 100, cmap = cmaps)
    cset1 = ax.contourf(x, y, rmse, 100, cmap = cmaps)
    plt.xlabel('$\it X$',fontsize=60)
    plt.ylabel('$\it Y$',fontsize=60)
    ax.set_xticks(np.linspace(0, 2*np.pi, 3),[r'$0$', r'$\pi$', r'$2\pi$'],size=40)
    ax.set_yticks(np.linspace(0, 2*np.pi, 3),[r'$0$', r'$\pi$', r'$2\pi$'],size=40)
    ax.set_aspect("equal")
    cbar = plt.colorbar(cset1)
    cbar.set_label('{}'.format(ob), size=60)
    cbar.ax.tick_params(labelsize=40)
    plt.show()      
    fig.savefig(path+'Temporal-averaged {} of '.format(ob) + method + ' solutions.png')
    plt.close('all')
    return

def get_levels(snapshots, num):
    maxz = 0
    minz = 0
    levels = 0
    if isinstance(snapshots, list):
        for i in range(len(snapshots)):
            if  max(snapshots[i]) > maxz:
                maxz = max(snapshots[i])
            if min(snapshots[i]) < minz:
                minz = min(snapshots[i])
    else:
        for i in range(np.shape(snapshots)[1]):
            if  max(snapshots[:,i]) > maxz:
                maxz = max(snapshots[:,i])
            if min(snapshots[:,i]) < minz:
                minz = min(snapshots[:,i])
    maxz = int(np.ceil(maxz))
    minz = int(np.floor(minz))
    levels = np.linspace(minz, maxz, num) 
    return maxz, minz, levels
