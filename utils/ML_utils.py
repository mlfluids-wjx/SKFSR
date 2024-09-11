import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import inv
from sklearn.metrics import mean_squared_error
import skimage
from skimage.metrics import structural_similarity as ssim
import math
import cv2

def fdn(Ds, x):
    x = np.squeeze(x, -1)
    N = []
    for d in Ds:
        count = 0
        for i in range(0, x.shape[0], d):
            for j in range(0, x.shape[1], d):
                for k in range(0, x.shape[2], d):
                    temp = x[i:i+d, j:j+d, k:k+d]
                    temp = np.any(temp)
                    count += temp
        N.append(count)
    N = np.stack(N)   
    return N

def fdnt(Ds, x):
    x = np.squeeze(x, 1)
    N = []
    for d in Ds:
        count = 0
        for i in range(0, x.shape[0], d):
            for j in range(0, x.shape[1], d):
                for k in range(0, x.shape[2], d):
                    temp = x[i:i+d, j:j+d, k:k+d]
                    temp = np.any(temp)
                    count += temp
        N.append(count)
    N = np.stack(N)   
    return N

def wrap_data(x_train, num_t):
    x_list = []
    lenth = len(x_train) - num_t
    for i in range(num_t):
        x_list.append(x_train[i:lenth+i+1])
    a = np.stack(x_list, axis=1)
    return a

def load_model(path):
    import torch
    model = torch.load(path)
    return model

def Norm_ev(w,vt,v):
    m = vt.T @ v
    l,u = scipy.linalg.lu(m, permute_l=True)
    print(l @ u == vt.T @ v)
    print(inv(l) @ m @ inv(u))
    vt = (inv(l) @ vt.T).T
    v = v @ inv(u)
    print(vt.T @ v)
    return vt, v

def Rec_func(Gx,vt,v):
    mode = v
    eig_func_list = []
    Gx_rec_list = []
    for i in range(Gx.shape[1]):
        eig_func = vt.T @ Gx[:,i,:]
        eig_func_list.append(eig_func)
        Gx_rec_list.append(mode @ eig_func)
    eig_func = np.stack(eig_func_list,axis=1)
    Gx_rec = np.stack(Gx_rec_list,axis=1)
    return eig_func, Gx_rec

def plot_latent(encoded, train_size, rank, path):
    plt.figure(figsize=(8,4), dpi=300)
    for i in range(rank):
        plt.plot(np.linspace(0,train_size,train_size), np.transpose(encoded[i,:]), label='latent dynamics {}'.format(i+1))
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)
    plt.legend(fontsize=12)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + 'latent dynamics' + '.jpg')
    return

def plot_trace(encoded, S_col, name):
    fig, ax = plt.subplots(figsize=(8,8), dpi=300, subplot_kw={"projection": "3d"})
    # cmap = plt.get_cmap("coolwarm")
    # from matplotlib.collections import LineCollection
    # dotColors = cmap(np.linspace(0,1,len(encoded[:,1])))
    ax.plot(*encoded[:3,:S_col], label='Reconstruction')
    ax.scatter(*encoded[:3,S_col:], label='Prediction')
    # ax.plot(encoded[:S_col,0], encoded[:S_col,1], encoded[:S_col,2], label='rec dynamics')
    # ax.plot(encoded[S_col:,0], encoded[S_col:,1], encoded[S_col:,2], label='pred dynamics', linestyle='dashed')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.set_zlabel(r'$y_3$')
    plt.tight_layout()
    plt.savefig(name)
    return
   
def plot_lstm(encoded, train_size, time, rank, path, name='lstm dynamics', label=False):
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    c = ['blue','orange','green','red','purple','brown','hotpink','aqua']
    for i in range(rank):
        ax.plot(np.array(time[:train_size]), np.transpose(encoded[i,:train_size]), 
                 label='latent dynamics {}'.format(i+1), c=c[i])
        try:
            ax.plot(np.array(time[train_size:]), np.transpose(encoded[i,train_size:]), ls='-.', 
                    label='predicted dynamics {}'.format(i+1), c=c[i])
        except:
            pass
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)

    ax.annotate('Prediction', xy=(1.05*time[train_size], np.min(encoded)),size=16,)
    ax.vlines(time[train_size], np.min(encoded), np.max(encoded), colors='black', ls='-')
    if label:
        # plt.legend(fontsize=16, loc='upper center', ncol=rank)
        for i in range(rank):
            # ax.annotate('{}'.format(i+1), xy=(time[0]), np.transpose(encoded[i,0]))
            ax.text(time[0]-0.5, np.transpose(encoded[i,0]),str(i+1),size=16,color=c[i])
    
    plt.title(name,size=24)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + name + '.jpg')
    return
    
def plotEncoded(encoded, train_size, pred_size, time, rank, path, name='lstm dynamics'):
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    c = ['blue','orange','green','red','purple','brown','hotpink','aqua']
    for i in range(rank):
        ax.plot(np.linspace(0,train_size,train_size), np.transpose(encoded[i,:train_size]), 
                 label='latent dynamics {}'.format(i+1), c=c[i])
        ax.plot(np.linspace(0,pred_size,pred_size), np.transpose(encoded[i,train_size:]), ls='-.', 
                 label='predicted dynamics {}'.format(i+1), c=c[i])
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)
    ax.annotate('Prediction', xy=(1.05*train_size, np.min(encoded)),size=16,)
    ax.vlines(train_size, np.min(encoded), np.max(encoded), colors='black', ls='-')
    # plt.legend(fontsize=12)
    plt.title(name,size=24)
    plt.xticks(ticks=time[:pred_size], size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + name)
    return

def plot_spatial_NME(snapshots, snapshots2, startT, trainT, endT, dltT, path, method, ob):
    rmse=np.zeros(len(snapshots))
    fig,ax = plt.subplots(figsize=(26, 16), dpi=300)
    for i in range(len(snapshots)):
        if ob == 'NME':
            rmse[i] = get_NME(snapshots[i], snapshots2[i])
        elif ob == 'RMSE':
            rmse[i] = get_rmse(snapshots[i], snapshots2[i])
        elif ob == 'NRMSE':
            rmse[i] = get_NRMSE(snapshots[i], snapshots2[i])
        elif ob == 'NRMSEsafe':
            rmse[i] = get_NRMSE_safe(snapshots[i], snapshots2[i])
        elif ob == 'SSIM':
            rmse[i] = get_ssim(snapshots[i], snapshots2[i])
        else:
            raise ValueError('one certain error method must be defined') 
        if rmse[i] == np.nan:
            rmse[i] = 0
    plt.plot(np.linspace(startT,endT,len(snapshots)),rmse, lw=3)
    plt.xlabel('Time (s)',fontsize=50)
    plt.ylabel('{}'.format(ob),fontsize=50)
    plt.xticks(size=40)
    plt.yticks(size=40)
    ax.yaxis.get_offset_text().set(size=30)
    ax.annotate('Reconstruction', xy=(0.6*trainT, min(rmse)),size=60,)
    ax.annotate('Prediction', xy=(1.05*trainT, min(rmse)),size=60,)
    ax.vlines(trainT, min(rmse), max(rmse), colors='black', ls='-.')
    plt.title(('\n Spatial-averaged {} of '.format(ob) + method + ' solutions \n'),size=60) 
    # plt.show()
    fig.savefig(path+'Spatial-averaged {} of '.format(ob) + method + ' solutions.png')  
    # fig.savefig(path+'Spatial-averaged {} of '.format(ob) + method + ' solutions.svg')  
    # with open(path+'rmse.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(rmse)
    fileObject = open(path+'{}.csv'.format(ob), 'w')  
    cc=0
    for ip in rmse:  
        fileObject.write(str(startT+dltT*cc)+',') 
        fileObject.write(str(ip))  
        fileObject.write('\n') 
        cc+=1
    fileObject.close()  
    return rmse

def video(start, end, pltT, dltT, path, time, ob):
    fps = 5
    file_path = path+'{}.mp4'.format(ob) # DIVX/mp4v
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi
    img = cv2.imread(path+ob+str(time[0])+'.jpg')    
    size=(img.shape[1],img.shape[0])
    videoWriter = cv2.VideoWriter(file_path,fourcc,fps,size, isColor=True)
    timesmall = time[:int((end-start)/dltT)]
    for num in timesmall[::pltT]:
        frame = cv2.imread(path+ob+str(num)+'.jpg')
        videoWriter.write(frame) 
    videoWriter.release()
    return

def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return mean_squared_error(records_real, records_predict)
    else:
        return None 

def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def get_NME(y_true, y_pred):
    NME = np.linalg.norm(y_true - y_pred, 2)
    NME /= np.linalg.norm(y_true, 2)
    return NME

def get_NRMSE(y_true, y_pred):
    NRMSE = get_rmse(y_true, y_pred)
    # NRMSE /= np.linalg.norm(y_true, 2)
    data_range = np.max(y_true) - np.min(y_true)
    NRMSE /= data_range
    # NRMSE /= len(y_true)
    return NRMSE

def get_ssim(y_true, y_pred):
    data_range = np.max(y_true) - np.min(y_true)
    ssim_value = ssim(y_true, y_pred, full=False, data_range=data_range)
    return ssim_value

def get_NRMSE_safe(y_true, y_pred):
    NRMSE_safe = np.linalg.norm(1 - (y_pred + 2)/(y_true + 2), 2)
    NRMSE_safe /= y_true.size
    # NRMSE_safe /= len(y_true)
    return NRMSE_safe

def get_tmporal_NME(snapshots, snapshots2, arr, ob):
    temporal_true=np.zeros((len(snapshots),len(arr)))
    temporal_predict=np.zeros((len(snapshots),len(arr)))
    rmse=np.zeros(len(arr))
    for i in range(len(snapshots)):
        for j in range(len(arr)):
            temporal_true[i][j] = snapshots[i][j]
            temporal_predict[i][j] = snapshots2[i][j]
    for i in range(len(arr)):
        if ob == 'NME':
            rmse[i] = get_NME(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'RMSE':
            rmse[i] = get_rmse(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'NRMSE':
            rmse[i] = get_NRMSE(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'NRMSEsafe':
            rmse[i] = get_NRMSE_safe(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'SSIM':
            rmse[i] = get_ssim(temporal_true[:,i], temporal_predict[:,i])
        else:
            raise ValueError('one certain error method must be defined') 
    return rmse

def sub_mean(S_full, S_mean):
    for i in range(S_full.shape[-1]):
        S_full[:,i] -= S_mean 
    return S_full

def plus_mean(S_full, S_mean):
    for i in range(S_full.shape[-1]):
        S_full[:,i] += S_mean 
    return S_full

def reshape(inp, numberx, numbery):
    return np.reshape(inp, (-1, numberx, numbery, 1))

def shape(inp, numberx, numbery):
    return np.reshape(inp, (numberx*numbery, -1))

def plot_loss(path, valid, method, loss_type='training'):
    import pandas as pd
    ml_head = pd.read_csv(path+'{}_log.csv'.format(loss_type), nrows=0)
    ml_head = list(ml_head)
    ml_loss = pd.read_csv(path+'{}_log.csv'.format(loss_type))
    ml_loss = np.array(ml_loss)
    epochs = ml_loss[:, 0]
    loss = ml_loss[:, 2]
    if type(valid) != type(None):
        loss_v = ml_loss[:, 4]  
        
    fig,ax = plt.subplots(figsize=(20, 16), dpi=300)
    plt.grid(True)
    plt.xticks(size=60)
    plt.yticks(size=60)
    plt.plot(epochs, loss, 'b', ls='--', lw=3, label='loss')
    if type(valid) != type(None):
        plt.plot(epochs, loss_v, 'r', ls='-.', lw=3, label='val loss')
    plt.xlabel('epoch',size=60)
    plt.ylabel('training loss',size=60)
    plt.legend(loc="upper right",fontsize=60)
    fig.savefig(path+'{} training loss.jpg'.format(method)) 
    return

def plot_loss2(path, method, loss_type='training'):
    import pandas as pd
    ml_head = pd.read_csv(path+'{}_log.csv'.format(loss_type), nrows=0)
    ml_head = list(ml_head)
    ml_loss = pd.read_csv(path+'{}_log.csv'.format(loss_type))
    ml_loss = np.array(ml_loss)
    epochs = ml_loss[:, 0]
    loss = ml_loss[:, 1]

    fig,ax = plt.subplots(figsize=(20, 16), dpi=300)
    plt.grid(True)
    plt.xticks(size=60)
    plt.yticks(size=60)
    plt.plot(epochs, loss, 'b', ls='--', lw=3, label='loss')

    plt.xlabel('epoch',size=60)
    plt.ylabel('training loss',size=60)
    plt.legend(loc="upper right",fontsize=60)
    fig.savefig(path+'{} training loss.jpg'.format(method)) 
    return

def plotEigs(
    eigs,
    show_axes=True,
    show_unit_circle=True,
    figsize=(8, 8),
    title="",
    dpi=None,
    filename=None,
):

    rank = eigs.size
    if dpi is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    plt.title('{} \n'.format(title), fontsize = 40)
    plt.gcf()
    ax = plt.gca()

    labellist = []     
    pointlist = []

    points, = ax.plot(
        eigs.real, eigs.imag, marker='o', 
        color='red', lw=0,
        )
    pointlist.append(points)
    labellist.append("Eigenvalues")  
    
    for i in range(rank):
        ax.annotate('{}'.format(i+1), xy=(eigs[i].real + 0.02, eigs[i].imag), )

        limit = 1.25*max(np.max(abs(eigs.real)),np.max(max(eigs.imag)),1)
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))
        
        # ax.set_xlim((-1, 2))
        # ax.set_ylim((-1, 2))   

        # x and y axes
        if show_axes:
            ax.annotate(
                "",
                xy=(np.max([limit * 0.8, 1.0]), 0.0),
                xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
                arrowprops=dict(arrowstyle="->"),
            )
            ax.annotate(
                "",
                xy=(0.0, np.max([limit * 0.8, 1.0])),
                xytext=(0.0, np.min([-limit * 0.8, -1.0])),
                arrowprops=dict(arrowstyle="->"),
            )

    plt.ylabel("Imaginary part", fontsize = 40)
    plt.xlabel("Real part", fontsize = 40)
    plt.xticks(size=24)
    plt.yticks(size=24)

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="green",
            fill=False,
            label="Unit circle",
            linestyle="--",
        )
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)

    ax.set_aspect("equal")
    a = pointlist
    a.append(unit_circle)
    b = labellist   
    b.append("Unit circle")
    ax.legend([pointlist,unit_circle],[labellist,"Unit circle"])
    ax.legend(a, b, loc = 'upper right', fontsize=20)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()   
    return

def plotDynamics(dynamics, time, title, path):
    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)     
    M = ['o', '^', 'v', '<', '>', 's', '*', '+', 'x', 'd', 'D']
    C = ['red', 'green', 'blue', 'orange', 'yellow', 'brown', 'purple', 'orange', 'cyan']
    L = ['-', '--', '.', '-.']
    
    i=0
    c=0
    for dynamic in dynamics:
        i+=1
        c+=1        
        if c >= len(M):
            c -= len(M) 
        plt.plot(time[:dynamic.shape[0]], dynamic.real, label='Dynamics {}'.format(i), marker=M[c-1],ms=15, lw=2)
    plt.xlabel('Time (s)', size=60)
    plt.ylabel('Dynamics', size=60)
    plt.xticks(size=40)
    plt.yticks(size=40)
    # ax.set_yscale('log')
    plt.title('\n {} \n'.format(title), fontsize=60)
    plt.legend(fontsize=32, loc="best")
    fig.savefig(path+'{}.jpg'.format(title))
    
    np.savetxt(path + '{}.csv'.format(title), dynamics, delimiter = ',')
    return

def plotSigma(sigma, rank_r=0.99, title='', filename=None):
    energy = np.cumsum(sigma**2 / (sigma**2).sum())
    if -1.0 < rank_r < 1.0:  
        rank = np.searchsorted(energy, rank_r) + 1
    elif rank_r == -1:
        rank = sigma.shape[0] - 1
    else:
        rank = rank_r - 1
    normal = sigma / np.cumsum(sigma)
    r = np.linspace(1, sigma.size, sigma.size)
    
        
    # r = range(energy.size)
    
    plt.figure(figsize=(16, 10), dpi=300) 
    plt.title('\n{}\n'.format(title), fontsize = 40)
    plt.gcf()
    ax = plt.gca()
    
    interval = 5
    
    for i in range(rank):
        ax.plot(
            r[i], normal[i], marker='o', 
            color='b', lw=0, ms=8,
            )
          
    for i in range(rank, sigma.size):
        ax.plot(
            r[i], normal[i], marker='o', 
            color='gray', lw=0, ms=5,
            )  
        
        # for i in range(0, rank-1, interval):
        #     ax.scatter(
        #         x=r[i], y=normal[i], marker='1', s=10,  
        #         color='red', 
        #         )
        #     ax.annotate('mode {}'.format(i+1), xy=(r[i]+2, normal[i]), )
            
    # ax.annotate('mode {}'.format(rank+1), xy=(r[rank]+2, normal[rank]), )
    ax.plot(
        r[rank], normal[rank], marker='o', 
        color='red', lw=0, ms=10,
        )  
    ax.annotate('mode {}'.format(rank+1), xy=(r[rank], normal[rank]), 
                xytext=(r[rank]+10, normal[rank]+0.2), arrowprops=dict(facecolor='black', shrink=0.1), size=30,)
        
    plt.ylabel(r"$\sigma_r$", fontsize = 40)
    plt.xlabel("rank", fontsize = 40)
    plt.xticks(size=24)
    plt.yticks(size=24)
    ax.set_yscale('log')
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()  

def compute_svd(X, svd_rank=0):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        beta = np.divide(*sorted(X.shape))
        tau = np.median(s) * omega(beta)
        rank = np.sum(s > tau)
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]
    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]
    return U, s, V

def txt_write(filename, ob):
    with open(filename, 'w') as f:
        f.write(str(ob))
        f.close
    return

class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False