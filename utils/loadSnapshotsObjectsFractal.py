import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable 
from matplotlib import colors

#----------------------------------Visualization---------------------------------#
def plotContour_fast(x, y, numberx, numbery, path, time, snapshots, cmaps, pltT, levels, minz, maxz, title, ob):
    norm1 = colors.Normalize(vmin=0, vmax=1, clip = True)
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
        fig, ax = plt.subplots(dpi=300)
        ax.set_title('\n {} field at {}s\n'.format(title, time[::pltT][i-1]),size=60) 
        # ss = np.reshape(snapshots[i], (numbery, numberx), order = 'C')
        ss = np.reshape(snapshots[::pltT][i-1], (numbery, numberx), order = 'C')
        plt.contourf(x, y, ss, 100, cmap = cmaps, norm=norm1)
        # cset1 = ax.contourf(x, y, ss, 100, cmap = cmaps, norm=norm1)
        cbar = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmaps), spacing='uniform')
        # cbar = plt.colorbar(cset1, spacing='uniform')
        # cbar = plt.colorbar()
        plt.clim(minz, maxz)
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)
        # cbar.set_ticks(levels)
        cbar.set_ticks([0,1])
        cbar.set_label('$\it {}$'.format(ob), size=60)
        cbar.ax.tick_params(labelsize=40)
        
        # plt.show()      
        fig.savefig(path+r'{} contour_{}.jpg'.format(title, time[::pltT][i-1]))
        plt.close('all')
    return


def plotModes_fast(x, y, numberx, numbery, rank, path, snapshot, cmaps, ob):
    for i in range(rank):
        # norm1 = colors.Normalize(vmin=minz, vmax=maxz, clip = True)
        ss = np.reshape(snapshot[:,i].real, (numbery, numberx), order = 'C')
        fig, ax = plt.subplots(figsize=(24, 8), dpi=300)
        ax.set_title('\n {} {}\n'.format(ob, i+1),size=60) 
        plt.contourf(x,y,ss,100, cmap = cmaps, extend="both")
        # cset1 = ax.contourf(x,y,ss,100, cmap = cmaps, extend="both")
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_xticks(np.linspace(0, 8*np.pi, 3),[r'$0$', r'$4\pi$', r'$8\pi$'],size=40)
        ax.set_yticks(np.linspace(-1, 1, 3),[r'$-1$', r'$0$', r'$1$'],size=40)
        # plt.clim(minz,maxz)
        cbar = plt.colorbar(cmap = cmaps, extend="both")
        # cbar.set_label('{}'.format(ob), size=30)
        ax.set_aspect(3)
        cbar.ax.tick_params(labelsize=40)
        cbar.ax.yaxis.get_offset_text().set(size=30)
        plt.show()      
        fig.savefig(path+r'{} contour_{}.jpg'.format(ob, i+1))
        plt.close('all')
    return

def plotOne_fast(x, y, S_full, cmaps, path=None, title=None, ob=None):
    fig=plt.figure(figsize = (x[0,-1] - x[0,0], y[-1,0] - y[0,0]))
    a = fig.add_subplot(121)
    a.set_axis_off()
    s = np.reshape(S_full, (x.shape[0], y.shape[1]), order = 'C')
    a.imshow(s, cmap=cmaps,
             extent = [x[0,0], x[0,-1] - x[0,0], y[0,0], y[-1,0] - y[0,0]],
             interpolation = 'none')
    if title != None:
        a.set_title('$\it {}$ contour \n'.format(title), size=x[0,-1] - x[0,0]) 
    if path != None:
        fig.savefig(path+r'{} contour.jpg'.format(ob))
    return

def plotContourErr_fast(x, y, numberx, numbery, path, time, snapshots, cmaps, pltT, levels, minz, maxz, ob):
    for i, snapshot in enumerate(snapshots[::pltT], start=1):
        ss = np.reshape(snapshots[i], (numbery, numberx), order = 'F')
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
    rmse = np.reshape(rmse, (numbery, numberx), order = 'C')
    fig, ax = plt.subplots(figsize=(24, 8), dpi=300)
    ax.set_title(('\n Temporal-averaged {} of '.format(ob) + method + ' solutions \n'),size=60) 
    plt.contourf(x, y, rmse, 100, cmap = cmaps)
    cset1 = ax.contourf(x, y, rmse, 100, cmap = cmaps)
    plt.xlabel('$\it X$',fontsize=60)
    plt.ylabel('$\it Y$',fontsize=60)
    ax.set_xticks(np.linspace(0, 8*np.pi, 3),[r'$0$', r'$4\pi$', r'$8\pi$'],size=40)
    ax.set_yticks(np.linspace(-1, 1, 3),[r'$-1$', r'$0$', r'$1$'],size=40)
    ax.set_aspect(3)
    cbar = plt.colorbar(cset1)
    cbar.set_label('{}'.format(ob), size=60)
    cbar.ax.tick_params(labelsize=40)
    plt.show()      
    fig.savefig(path+'Temporal-averaged {} of '.format(ob) + method + ' solutions.png')
    plt.close('all')
    return

def get_levels(snapshots, num):
    maxz = float('%.2f' % np.max(snapshots))
    minz = float('%.2f' % np.min(snapshots))
    maxz = min(abs(minz), abs(maxz))
    minz = -maxz
    levels = np.linspace(minz, maxz, num)
    for i in range(levels.size):
        levels[i] = float('%.2f' % levels[i])
    return maxz, minz, levels
