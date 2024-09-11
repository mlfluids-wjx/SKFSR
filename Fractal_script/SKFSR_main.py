import sys
sys.path.append('../utils')
sys.path.append('../Flow_scripts')
import numpy as np
import matplotlib.pyplot as plt
import generator as gt
from sklearn import preprocessing
import torch
import time as TT
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd 
import datetime
import os
import ML_utils as ut
from scipy import interpolate
import argparse
import fdgpu as f
torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='method', type=str,
                    default='Super_SKFSR_single',
                    help=' arguments:  binarization + training strategy + snapshot quantity'  
                    'DB/ TB + unsp/ semi/ sup + single/ few/ standard'
                    ' eg. Super_SKFSR_DB_unsp_single')
parser.add_argument('-c', dest='case', type=str, default='Channel', help=
                    'Name of Dataset: Channel/ Iso/ Koch126/'
                    ' eye/ Koch15/ Koch146/ Sire189/ ')
parser.add_argument('-b', dest='batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('-s', dest='scaler', type=int, default=1, help="Super Resolution Ratio X2")
parser.add_argument('-e',dest='epochs', type=int, default=1000, help="Number of Epochs")
parser.add_argument('--load', dest='load', default=False, action='store_true')
args = parser.parse_args()

method = args.method
case = args.case
batch_size = args.batch_size
epochs = args.epochs
load = args.load
zoom = args.scaler

save_steps = 20
patience = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '../../output/Fractal_{}_{}/'.format(case, method)
acitv = 'relu'
t = 32
Lmin = 1
bins = 10
cmap = plt.cm.PuBuGn
xx = 512
normalize = True
oom = False
if 'unnorm' in method:
    normalize = False
    
if zoom != 1 or '4X' in method or '8X' in method:
    if '4X' in method:
        zoom = 2
    elif '8X' in method:
        zoom = 4
        
if 'avg' in method:     
    sub = nn.AvgPool2d(2*zoom)
else:
    sub = nn.MaxPool2d(2*zoom)

if not os.path.exists(path):
    os.makedirs(path) 
    
dltT = 1
S_col = 1
S_col_full = 1
pltT = int(1)      
startT = 0 

# Fractal / turbulence cases
if case == 'Koch126':
    import loadSnapshotsObjectsFractal as ls
    pms = np.load('../data/koch126_512_9.npy').astype(np.float32)
    pms = np.expand_dims(pms, axis=0)
    pms = np.expand_dims(pms, axis=0)
    pm = sub(torch.tensor(pms))
    pm = pm.clone().detach().cpu().numpy()
    numberxs, numberys = xx, xx
    x = np.linspace(0, xx, numberxs)
    y = np.linspace(0, xx, numberys)
    ob = ''
    
elif case == 'eye':
    import loadSnapshotsObjectsFractal as ls
    pms = np.eye(xx).astype(np.float32)
    pms = np.expand_dims(pms, axis=0)
    pms = np.expand_dims(pms, axis=0)
    pm = sub(torch.tensor(pms))
    pm = pm.clone().detach().cpu().numpy()
    numberxs, numberys = xx, xx
    x = np.linspace(0, xx, numberxs)
    y = np.linspace(0, xx, numberys)
    ob = ''

elif case == 'Koch15':
    import loadSnapshotsObjectsFractal as ls
    pms = np.load('../data/koch15_512_7.npy').astype(np.float32)
    pms = np.expand_dims(pms, axis=0)
    pms = np.expand_dims(pms, axis=0)
    pm = sub(torch.tensor(pms))
    pm = pm.clone().detach().cpu().numpy()
    numberxs, numberys = xx, xx
    x = np.linspace(0, xx, numberxs)
    y = np.linspace(0, xx, numberys)
    ob = ''

elif case == 'Koch146':
    import loadSnapshotsObjectsFractal as ls
    pms = np.load('../data/koch146_512_9.npy').astype(np.float32)
    pms = np.expand_dims(pms, axis=0)
    pms = np.expand_dims(pms, axis=0)
    pm = sub(torch.tensor(pms))
    pm = pm.clone().detach().cpu().numpy()
    numberxs, numberys = xx, xx
    x = np.linspace(0, xx, numberxs)
    y = np.linspace(0, xx, numberys)
    ob = ''


elif case == 'Sier189':
    import loadSnapshotsObjectsFractal as ls
    pms = gt.sierpinski_foam(dmin=1, n=6, ndim=2, max_size=1e9)
    pms = np.float32(pms)[108:-109,108:-109]
    numberxs, numberys = xx, xx
    x = np.linspace(0, numberxs, numberxs)
    y = np.linspace(0, numberys, numberys)
    pms = np.expand_dims(pms, axis=0)
    pms = np.expand_dims(pms, axis=0)
    pm = sub(torch.tensor(pms))
    pm = pm.clone().detach().cpu().numpy()
    ob = ''
    
elif 'Channel' in case:
    import loadSnapshotsObjectsChannel as ls
    filename = '../data/channel_u_512X128p_4to24s_dt0.13.npy'
    numberxs, numberys = 512, 128
    ob = 'u'
    pms = np.load(filename)[:151,:,:,0:1]
    pms = pms.transpose(0,3,2,1)
    pm = torch.tensor(pms)
    pm = sub(pm)
    pm = pm.clone().detach().cpu().numpy()
    dltT = 0.13
    S_col = 120
    S_col_full = 151
    pltT = int(10)      
    startT = 4 
    
    x = np.linspace(0, 8*np.pi, numberxs)
    y = np.linspace(-1, 1, numberys)
    
    cmap = plt.cm.jet
    
elif 'Iso' in case:
    import loadSnapshotsObjectsJHTDB as ls
    filename = '../data/iso_velocity_pressure_256x256_2to8.npy'
    numberxs, numberys = 256, 256
    ob = 'u'
    pms = np.load(filename)[:151,:,:,0:1]
    pms = pms.transpose(0,3,2,1)
    pm = torch.tensor(pms)
    pm = sub(pm)
    pm = pm.clone().detach().cpu().numpy()
    dltT = 0.03
    S_col = 120
    S_col_full = 151
    pltT = int(10)      
    startT = 2 
    
    x = np.linspace(0, 2*np.pi, numberxs)
    y = np.linspace(0, 2*np.pi, numberys)
    
    cmap = plt.cm.jet
else:
    raise ValueError('no case selected')
    
if 'single' in method:
    S_col = 30
if 'few' in method:
    S_col = 1

plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'
font={'family': 'Times New Roman', 'math_fontfamily':'stix'}


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass
    
numberx, numbery = pm.shape[-2:]
numberxs, numberys = pms.shape[-2:]
c = pms.shape[1]
xc, yc = x[::2*zoom], y[::2*zoom]

timec_start = TT.perf_counter()
pmc = []
for s in range(c):
    pmcc = []
    for i in range(pm.shape[0]):
        cub = interpolate.RectBivariateSpline(np.arange(0, numberx), np.arange(0, numbery), pm[i,s])
        cc = cub(np.linspace(0, numberx, numberxs), np.linspace(0, numbery, numberys))
        pmcc.append(np.reshape(cc, (-1,numberxs,numberys), order='F'))
    pmc.append(np.array(pmcc))
pmc = np.concatenate(pmc, axis=1)
snapshots3 = list(np.reshape(pmc, (-1, numberxs*numberys), order='F'))
timec_end = TT.perf_counter()
pmct = torch.tensor(pmc).to(device)

if normalize:
    if 'Iso' in path:
        scaler = preprocessing.MaxAbsScaler()
        scalers = preprocessing.MaxAbsScaler()
    else:
        scaler = preprocessing.MinMaxScaler()
        scalers = preprocessing.MinMaxScaler()
    pm = np.reshape(pm, (-1, numberx*numbery), order='F')
    pms = np.reshape(pms, (-1, numberxs*numberys), order='F')  
    pm = scaler.fit_transform(pm)  
    pms = scalers.fit_transform(pms)

pm = np.reshape(pm, (-1,c,numberx,numbery), order='F')
pms = np.reshape(pms, (-1,c,numberxs,numberys), order='F')

Dmin = np.ceil(Lmin * min(numberx, numbery))
mm = np.logspace(0, 9, 10, endpoint=True, base=2)
Ds = []
for i in mm:
    if i <= Dmin:
        Ds.append(int(i))
Ds = np.array(Ds)

Ds = np.unique([np.round(i).astype(int) for i in Ds]) 
Ds2 = Ds*2*zoom
Dsc = Ds2
Dsc = np.concatenate([[Ds[_] for _ in range(zoom)], Ds2])
Dsc = np.unique([np.round(_).astype(int) for _ in Dsc]) 
bina = f.binarize().to(device)
if 'DB' in path:
    binan = f.binarizen().to(device)
    if 'Iso' in path:
        binan = f.binarizen(e=-1).to(device)
else:
    binan = f.binarize().to(device)

fdf = f.fd2df().to(device)

if 'avg' in path:
    fda = f.fd2da().to(device)
else:
    fda = fdf

pmt = torch.tensor(pm).to(device)
pmst = torch.tensor(pms).to(device)
meanst = pmst.mean()
meant = pmt.mean()
pmb = bina(pmt, meant)
pmsb = bina(pmst, meanst)
pmcb = bina(pmct)

Nf = np.array([fda(binan(pmst[:S_col], meanst), Dsc[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Dsc))])
N = np.array([fda(binan(pmt[:S_col], meant), Ds[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Ds))])
Ns = np.array([fda(binan(pmst[:S_col], meanst), Ds[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Ds))])
Nsc = np.array([fda(binan(pmst[:S_col], meanst), Ds2[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Ds2))])

Fd_s, Fbs = - np.polyfit(np.log(Ds), np.log(Ns), 1)
Fd_sc, Fbsc = - np.polyfit(np.log(Ds2), np.log(Nsc), 1)

if not 'semi' in path:
    Nrec = np.vstack([np.float32(np.round(
        np.exp(-Fd_sc*np.log(Dsc[_])-Fbs))) for _ in range(zoom)])
else:
    Nrec = np.round(Ns[0:zoom])
Nsc = np.concatenate([Nrec, Nsc])

if 'FNO' in method:
    from neuralop.models import TFNO
    model = TFNO(n_modes=(numberx, numbery), hidden_channels=32, in_channels=1,
                 projection_channels=64, factorization='tucker', rank=0.42)
    model = model.to(device)
elif 'UNet' in method:
    import ML_UNet_TorchModel as m
    model = m.UNet().to(device)
elif 'SKFSR' in method:
    import ML_SKFSR_TorchModel as m
    model = m.model(scaler=zoom)
    model = model.to(device)
elif 'ResNet' in method:
    from torchvision import models
    model = models.resnet101(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, numberx*numbery)
    model = model.to(device)
  
# Upscaler
class ExtendedModel(nn.Module):
    def __init__(self):
        super(ExtendedModel, self).__init__()
        self.base_model = model
        self.convu = nn.Conv2d(1, 2*(zoom*2)**2, 3, padding='same')
        self.conve = nn.Conv2d(2, 1, 3, padding='same')
        self.t = nn.ReLU(True)
        self.ps = nn.PixelShuffle(zoom*2)
        self.upsamp = nn.Upsample(scale_factor=(2,2))

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(-1, c, numberx, numbery)
        x = self.convu(x)
        x = self.t(x)
        x = self.ps(x)
        x = self.conve(x)
        return x

if not 'SKFSR' in method:
    model2 = ExtendedModel()
    del model, ExtendedModel
    model = model2
    del model2
    model = model.to(device)

# Read data
X_train = TensorDataset(pmt[:S_col], torch.tensor(Nrec[:,:S_col].T), pmst[:S_col])
X_train = DataLoader(
    X_train, batch_size=batch_size, shuffle=False, num_workers=0)

# Training
offlineStart = TT.perf_counter()

if load:
    print('load trained model...')
    model.load_state_dict(torch.load(path+'ckpt.pkl'))
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_steps = 0
    best_loss = float(np.Inf)
    if not (patience==None or patience==0):
        early_stopper = ut.EarlyStopper(patience=patience, min_delta=0)
    
    dfhistory = pd.DataFrame(columns = ["epoch","loss","loss_rec","loss_fpred","loss_npred"]) 
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)
    
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        loss_total = .0
        
        for step, (x0, N0, x1) in enumerate(X_train, 1): 
            x0 = x0.to(device)
            x1 = x1.to(device)
            N0 = N0.to(device)
            
            optimizer.zero_grad()
            y1 = model(x0)
            
            l_pred = loss_fn(x1, y1)
             
            xm0 = binan(x0, meant)
            xm1 = binan(x1, meanst)
            if 'DB' in path:
                if '4X' in path or '8X' in path:
                    ym1 = binan(y1)
                else:
                    ym1 = binan(y1, meanst)
            else:
                ym1 = binan(y1, meanst)
            
            lf_pred = .0
            ln_pred = .0

            if 'semi' in path:
                xm1 = bina(x1)
                for s in range(len(Dsc)):
                    xf1 = fdf(x1, Dsc[s])
                    xn1 = fda(xm1, Dsc[s])

                    yf1 = fdf(y1, Dsc[s])
                    yn1 = fda(ym1, Dsc[s])

                    lf_pred += loss_fn(xf1, yf1)/len(Dsc)
                    ln_pred += 0.1*loss_fn(xn1, yn1)/len(Dsc)

            else:
                for s in range(len(Ds)):
                    xf0 = fdf(x0, Ds[s])
                    xn0 = fda(xm0, Ds[s]).mean(axis=[1,2,3])

                    yf1 = fdf(y1, Ds2[s])
                    yn1 = fda(ym1, Ds2[s]).mean(axis=[1,2,3])

                    lf_pred += loss_fn(xf0, yf1)/len(Ds)
                    ln_pred += 0.01*loss_fn(xn0, yn1)/len(Dsc)
                lf_pred += loss_fn(nn.MaxPool2d(2*zoom)(y1), x0)
                
                for b in range(zoom):
                    ln_pred += 0.01*loss_fn(N0[:,b]/(numberxs*numberys), 
                                fda(ym1, Dsc[b]).mean(axis=[1,2,3])
                                )/len(Dsc)
                
            if 'sup' in path:
                loss = l_pred
            else:
                loss = lf_pred + ln_pred
                  
            loss.backward()
    
            loss_total += loss.item()
            optimizer.step()
            
            lr = optimizer.param_groups[0]["lr"]
            
            loss_pred = l_pred.item()
            loss_fpred = lf_pred.item()
            loss_npred = ln_pred.item()

            
            info = (epoch, loss_total/step, loss_pred, loss_fpred, loss_npred)
            dfhistory.loc[epoch-1] = info
            print(("[step = %d] loss: %.3f,") % (step, loss_total/step))
            print(("\nEPOCH = %d, loss = %.5f," " loss_rec = %.5f, " 
                   " loss_fpred = %.5f, " " loss_npred = %.5f, " ) %info)
         
        if 'time' not in path or save_steps == 0:
            if loss_total < best_loss:
                best_steps += 1
                if best_steps % save_steps == 0:
                    best_loss = loss_total
                    print('------Save model:best loss is %.3f------' %best_loss)
                    torch.save(model.state_dict(), path+'ckpt.pkl')
            
        if not (patience==None or patience==0):
            if early_stopper.early_stop(loss_total):       
                print('------Early stopping------')
                break
    
        print("\n"+"=========="*6 + "%s"%nowtime)
        
    print('Finished Training...')
    offlineEnd = TT.perf_counter()
    dfhistory.to_csv(path+'training_log.csv', sep=',')
    try:
        model.load_state_dict(torch.load(path+'ckpt.pkl'))
    except:
        print('ckpt not read')
del X_train, patience, pmst, pmb, pmsb, pmct

if not 'SKFSR' in path:
    try:
        os.remove(path+'ckpt.pkl')
    except:
        print('ckpt not read')

torch.cuda.empty_cache
model.eval() 
onlineStart = TT.perf_counter()
with torch.no_grad():
    try:
        pm2_t = model(pmt)
        pm2 = pm2_t.clone().detach().cpu().numpy()
    except:    
        oom = True
        torch.cuda.empty_cache
        pm2_t = np.concatenate([model(pmt[i:i+1]).clone().detach().cpu().numpy(
            ) for i in range(len(pmt))])       
onlineEnd = TT.perf_counter()

if normalize:
    pm = np.reshape(pm, (-1, numberx*numbery), order='F')
    pm = scaler.inverse_transform(pm)
    pm = np.reshape(pm, (-1,1,numberx,numbery), order='F')
    pm2 = np.reshape(pm2, (-1, numberxs*numberys), order='F')
    pm2 = scalers.inverse_transform(pm2)
    pm2 = np.reshape(pm2, (-1,1,numberxs,numberys), order='F')
    pms = np.reshape(pms, (-1, numberxs*numberys), order='F')
    pms = scalers.inverse_transform(pms)
    pms = np.reshape(pms, (-1,1,numberxs,numberys), order='F')
    
pmt = torch.tensor(pm) 
pmb = bina(pmt)

pm2_t = torch.tensor(pm2).to(device)
pm2b = bina(pm2_t)

pmst = torch.tensor(pms).to(device)
pmsb = bina(pmst)

np.save(path+'/data_sr', pm2)

trainT = startT + S_col * dltT
endT = startT + S_col_full * dltT

maxz = float('%.2f' % np.max(pms))
minz = float('%.2f' % np.min(pms))
levels = np.linspace(minz, maxz, 11)
time = []
time_str = []
for i in np.arange(startT, endT, dltT):
    i = float('%.3f' % i)
    time.append(i)
    time_str.append(str('%.3f' % i))
else:
    time_n = time
for i in range(levels.size):
    levels[i] = float('%.2f' % levels[i])
snapshots = list(np.reshape(pms, (-1, numberxs*numberys), order='F'))
snapshots2 = list(np.reshape(pm2, (-1, numberxs*numberys), order='F'))
snapshotsc = list(np.reshape(pm, (-1, numberx*numbery), order='F'))
xi,yi = np.meshgrid(x,y)
xic,yic = np.meshgrid(xc,yc)

minT = int(min(len(snapshots), len(snapshots2), len(snapshots3), len(snapshotsc)))    
snapshots = snapshots[:minT]
snapshots2 = snapshots2[:minT]
snapshots3 = snapshots3[:minT]
snapshotsc = snapshotsc[:minT]

try:
    spatial_RMSE = ut.plot_spatial_NME(snapshots, snapshots2, time[0],
                                       trainT, time[minT-1], dltT, path, 
                                       method='{}'.format(method), ob='RMSE')
    
    spatial_RMSE2 = ut.plot_spatial_NME(snapshots, snapshots3, time[0],
                                       trainT, time[minT-1], dltT, path, 
                                       method='{}'.format('cubic'), ob='RMSE')
except:
    print('input contains Nan')

if not 'rigin' in path:
    ls.plotContour_fast(xi, yi, numberxs, numberys, path, time_str, snapshots2, cmap, pltT, 
                    levels, minz, maxz, title='{} reconstructed'.format(method), ob=ob)

if 'rigin' in path:
    ls.plotContour_fast(xi, yi, numberxs, numberys, path, time_str, snapshots, cmap, pltT, 
                    levels, minz, maxz, title='Original', ob=ob)
    ls.plotContour_fast(xic, yic, numberx, numbery, path, time_str, snapshotsc, cmap, pltT, 
                    levels, minz, maxz, title='Input', ob=ob)
    ls.plotContour_fast(xi, yi, numberxs, numberys, path, time_str, snapshots3, cmap, pltT, 
                    levels, minz, maxz, title='cubic', ob=ob)

try:
    if not 'rigin' in path:
        ut.video(startT, endT, pltT, dltT, path, time_str, ob='{} reconstructed_'.format(method))
    if 'rigin' in path:
        ut.video(startT, endT, pltT, dltT, path, time_str, ob='Input_')
        ut.video(startT, endT, pltT, dltT, path, time_str, ob='cubic_')
        ut.video(startT, endT, pltT, dltT, path, time_str, ob='Original_')
except:
    print('animation skipped')


RMSE_train = ut.get_rmse(snapshots[:S_col], snapshots2[:S_col])
RMSE_train2 = ut.get_rmse(snapshots[:S_col], snapshots3[:S_col])
if minT > S_col:
    RMSE_pred = ut.get_rmse(snapshots[S_col:], snapshots2[S_col:])
    RMSE_all = ut.get_rmse(snapshots, snapshots2)
    RMSE_pred2 = ut.get_rmse(snapshots[S_col:], snapshots3[S_col:])
    RMSE_all2 = ut.get_rmse(snapshots, snapshots3)
else:
    RMSE_pred = np.nan
    RMSE_all = np.nan
    RMSE_pred2 = np.nan
    RMSE_all2 = np.nan
RMSE_list = [RMSE_train, RMSE_pred, RMSE_all]
RMSE_list2 = [RMSE_train2, RMSE_pred2, RMSE_all2]

loc = locals()
def get_variable_name(variable):
    for k,v in loc.items():
        if loc[k] is variable:
            return k
        
def write_NME(RMSE_list, writepath, ob):
    fileObject = open(writepath+'/{}_list.txt'.format(ob), 'w')  
    for i in range(len(RMSE_list)): 
        fileObject.write(get_variable_name(RMSE_list[i])+'\r') 
        fileObject.write(str('%.6f' % RMSE_list[i]) ) 
        fileObject.write('\n') 
    fileObject.close()  

write_NME(RMSE_list, path, 'RMSE')
write_NME(RMSE_list2, path, 'RMSE_cubic')

Nf = np.array([fdf(pmsb, Dsc[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Dsc))])
N2 = np.array([fdf(pm2b, Ds[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Ds))])
N2f = np.array([fdf(pm2b, Dsc[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Dsc))])
Ncubic = np.array([fdf(pmcb, Ds[i]).sum(axis=[1,2,3]).clone().detach().cpu().numpy() for i in range(len(Ds))])

np.savetxt(path+'/N2.csv', np.int32(N2), delimiter=',', fmt="%d")
np.savetxt(path+'/N2_full.csv', np.int32(N2f), delimiter=',', fmt="%d")
np.savetxt(path+'/Ncubic.csv', np.int32(Ncubic), delimiter=',', fmt="%d")

F_f, ___ = - np.polyfit(np.log(Dsc), np.log(Nf), 1)
F_2f, ___ = - np.polyfit(np.log(Dsc), np.log(N2f), 1)
Fd_c, __ = - np.polyfit(np.log(Ds), np.log(Ncubic), 1)

np.savetxt(path+'/F_f.txt', F_f)
np.savetxt(path+'/F_2f.txt', F_2f)
np.savetxt(path+'/Fd_c.txt', Fd_c)

if not load:
    file = open(path+'time.txt', 'w')
    offlineTime = offlineEnd - offlineStart
    onlineTime = onlineEnd - onlineStart
    cubicTime = timec_end - timec_start
    time_per_epoch = offlineTime / epoch
    print('offline time: '+ str('%.5f' % offlineTime))
    print('online time: '+ str('%.5f' % onlineTime))
    print('time per epoch: '+ str('%.5f' % time_per_epoch) + 's')
    print('RMSE: ' + str('%.5f' % RMSE_train) +' / '+ str('%.5f' % RMSE_pred))
    print('RMSE cubic: ' + str('%.5f' % RMSE_train2) +' / '+ str('%.5f' % RMSE_pred2))
    file.write('offline time: '+ str('%.5f' % offlineTime) + 's' + '\n')
    file.write('online time: '+ str('%.5f' % onlineTime) +'s' + '\n')
    file.write('time per epoch: '+ str('%.5f' % time_per_epoch) +'s' + '\n')
    file.write('cubic time: '+ str('%.5f' % cubicTime) +'s' + '\n')
    file.write('CUDA OOM: '+str(oom))
    file.close()
  
F_train, F_pred, F_all = F_f[:S_col].mean(), F_f[S_col:].mean(), F_f.mean()
F2_train, F2_pred, F2_all = F_2f[:S_col].mean(), F_2f[S_col:].mean(), F_2f.mean()
Fc_train, Fc_pred, Fc_all = Fd_c[:S_col].mean(), Fd_c[S_col:].mean(), Fd_c.mean()
Fd_list = [F_train, F_pred, F_all]
Fd2_list = [F2_train, F2_pred, F2_all]
Fdc_list = [Fc_train, Fc_pred, Fc_all]
write_NME(Fd_list, path, 'F_f')
write_NME(Fd2_list, path, 'F_2f')
write_NME(Fdc_list, path, 'Fd_c')

torch.cuda.empty_cache 
print('path =' , path)
print('finish')
