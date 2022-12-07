from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import time
#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        

    def __len__(self):

        return len(self.D)
        

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist, hist_refPos = self.getHistory(vehId,t,vehId,dsId)  # hist = [16, 2]-----历史轨迹点16个
        fut, fut_refPos = self.getFuture(vehId,t,dsId)  # fut = [25, 2]-----预测轨迹点25个

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            tmp, _ = self.getHistory(i.astype(int), t,vehId,dsId)
            neighbors.append(tmp)
        # neighbors = list [39, *, *]
        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist,fut,neighbors,lat_enc,lon_enc,hist_refPos,fut_refPos



    ## Helper function to get track history
    # 车辆id 帧号 车辆id 场景id
    # refVeh被预测的车
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2]), None
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2]), None
            
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2]), None
            else:
                # 取当前帧以前的30个点和当前帧后一个点，然后2个点向下采样，所以历史轨迹为16个点
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2]), None
            return hist, refPos



    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        # 取当前帧以后的50个点和当前帧后一个采样点，然后2个点向下采样，所以未来轨迹为25个点
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut, refPos
        


    ## Collate function for dataloader
    def collate_fn(self, samples):
        # samples = [128, 7] ----- [batch, [hist,fut,neighbors,lat_enc,lon_enc,hist_refPos,fut_refPos]]
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_,_,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)


        # Initialize social mask batch:
        # ??? 和interaction有关
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.byte()


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        hist_refPos_batch = torch.zeros(len(samples), 2)
        fut_refPos_batch = torch.zeros(len(samples), 2)
        # hist_batch = [16, 128, 2]
        # fut_batch = [25, 128, 2]
        # op_mask_batch = [16, 128, 2]
        # lat_enc_batch = [128, 3]
        # lon_enc_batch = [128, 2]
        # hist_refPos_batch = [128, 2]
        # fut_refPos_batch = [128, 2]

        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc, hist_refPos, fut_refPos) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            hist_refPos_batch[sampleId,:] = torch.from_numpy(hist_refPos)
            fut_refPos_batch[sampleId, :] = torch.from_numpy(fut_refPos)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    # print(len(nbr))
                    # print(nbr)

                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1
        # print(count)
        # sys.exit()
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, hist_refPos_batch ,fut_refPos_batch

#________________________________________________________________________________________________________________________________________





## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]  # muX.shape = [25, 128, 1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]  # 相关系数
    sigX = torch.exp(sigX)  # 0-inf
    sigY = torch.exp(sigY)  # 0-inf
    rho = torch.tanh(rho)  # -1-1
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)  # out.shape = [25, 128, 5]
    return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1):
    # out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):
    out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160

    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                # If we represent likelihood in feet^(-1):
                # out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                # If we represent likelihood in m^(-1):
                out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        # out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def caluate_uncertainty(deep_ensemble, select=None):
    if select != None:
        deep_ensemble = deep_ensemble[:, :, select, :]
        ensemble_mux = torch.mean(deep_ensemble[:, :, 0], dim=0)
        ensemble_muy = torch.mean(deep_ensemble[:, :, 1], dim=0)
        
        # print('ensemble_muy:')
        # print(ensemble_muy*0.3048)
        
        
        mux = deep_ensemble[:, :, 0]
        muy = deep_ensemble[:, :, 1]
        sigmax = 1 / deep_ensemble[:, :, 2]
        sigmay = 1 / deep_ensemble[:, :, 3]
        p = deep_ensemble[:, :, 4]
        # print('mux:')
        # print(mux*0.3048)
        
        # print('muy:')
        # print(muy*0.3048)
        
        # ensemble_plot(mux, muy)
        
        uncertainty_vx = torch.mean(torch.pow(sigmax, 2) + torch.pow(mux, 2), dim=0) - torch.pow(ensemble_mux, 2)
        uncertainty_vy = torch.mean(torch.pow(sigmay, 2) + torch.pow(muy, 2), dim=0) - torch.pow(ensemble_muy, 2)
        uncertainty_cv = torch.mean(p*sigmax*sigmay + mux*muy, dim=0) - ensemble_mux * ensemble_muy
        
        
        
        return ensemble_mux, ensemble_muy, uncertainty_vx, uncertainty_cv, uncertainty_cv, uncertainty_vy 



def ensemble_plot(ensemble_x, ensemble_y, x, y, hx, hy):
    plt.figure(1)
    
    
    
    # ensemble plot
    for i in range(ensemble_x.shape[0]):
        plt.scatter(ensemble_y[i, :].detach().cpu().numpy()*0.3048, ensemble_x[i, :].detach().cpu().numpy()*0.3048, s=1, c='midnightblue', label=i)

    # plt.scatter(ensemble_y[1, :].detach().cpu().numpy()*0.3048, ensemble_x[1, :].detach().cpu().numpy()*0.3048, s=1, c='navy', label=1)
    # plt.scatter(ensemble_y[2, :].detach().cpu().numpy()*0.3048, ensemble_x[2, :].detach().cpu().numpy()*0.3048, s=1, c='darkblue', label=2)
    # plt.scatter(ensemble_y[3, :].detach().cpu().numpy()*0.3048, ensemble_x[3, :].detach().cpu().numpy()*0.3048, s=1, c='mediumblue', label=3)
    # plt.scatter(ensemble_y[4, :].detach().cpu().numpy()*0.3048, ensemble_x[4, :].detach().cpu().numpy()*0.3048, s=1, c='blue', label=4)
    # plt.scatter(ensemble_y[5, :].detach().cpu().numpy()*0.3048, ensemble_x[5, :].detach().cpu().numpy()*0.3048, s=1, c='blue', label=5)
    

    ensemble_mux = torch.mean(ensemble_x[:, :], dim=0)
    ensemble_muy = torch.mean(ensemble_y[:, :], dim=0)
    plt.scatter(ensemble_muy.detach().cpu().numpy()*0.3048, ensemble_mux.detach().cpu().numpy()*0.3048, s=2, c='aqua', label='mu')
    plt.scatter(y, x, s=1, c='r')
    plt.scatter(hy, hx, s=1, c='g')
    
    limitH = max(ensemble_x[0, :].detach().cpu().numpy()*0.3048)
    limitL = min(ensemble_x[0, :].detach().cpu().numpy()*0.3048)
    for i in range(ensemble_x.shape[0]):
        limitH = max(max(ensemble_x[i, :].detach().cpu().numpy()*0.3048), limitH)
        limitL = min(min(ensemble_x[i, :].detach().cpu().numpy()*0.3048), limitL)

    
    limitH = max(max(limitH, max(hx)), max(x))
    limitL = min(min(limitL, min(hx)), min(x))
    
    plt.ylim(limitL-5, limitH+5)

    plt.legend()
    plt.savefig('eval_unc')
    plt.cla()
    # time.sleep(0.1)