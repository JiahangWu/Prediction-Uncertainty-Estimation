from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest, caluate_uncertainty, ensemble_plot
from torch.utils.data import DataLoader
import time
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ## Network Arguments
    args = {}
    args['use_cuda'] = True
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13,3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = False
    args['train_flag'] = False
    args['ensemble_size'] = 5


    # Evaluation metric:
    # metric = 'nll'  #or rmse
    metric = 'rmse'  #or rmse
    batch_size = 128
    # Deep-ensemble Initialize network
    nets = [highwayNet(args) for ensemble in range(args['ensemble_size'])]
    models = ['trained_models/cslstm_m0.tar',
                'trained_models/cslstm_m1.tar',
                'trained_models/cslstm_m2.tar',
                'trained_models/cslstm_m3.tar',
                'trained_models/cslstm_m4.tar']
    for i, net in enumerate(nets):
        net.load_state_dict(torch.load(models[i]))
        # net.eval()
        if args['use_cuda']:
            net = net.cuda()

    trSet = ngsimDataset('data/TrainSet.mat')
    valSet = ngsimDataset('data/ValSet.mat')
    tsSet = ngsimDataset('data/TestSet.mat')
    trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=False,num_workers=1,collate_fn=trSet.collate_fn)  # 使用collate_fun可以自己操作每个batch的数据
    valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=False,num_workers=1,collate_fn=valSet.collate_fn)
    tsDataloader = DataLoader(tsSet,batch_size=batch_size,shuffle=False,num_workers=1,collate_fn=tsSet.collate_fn)

    lossVals = torch.zeros(25).cuda()
    counts = torch.zeros(25).cuda()
    
    print(metric)

    for i, data in enumerate(trDataloader):
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, hist_refPos, fut_refPos = data

        # Initialize Variables
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            hist_refPos = hist_refPos.cuda()
            fut_refPos = fut_refPos.cuda()

        # Deep ensemble
        # deep_ensemble = [ensemble_size, out_length=25, batch_size, 5]
        deep_ensemble = torch.zeros((args['ensemble_size'], args['out_length'], batch_size, 5), dtype=torch.double)
        if args['use_cuda']:
            deep_ensemble = deep_ensemble.cuda()
            
        for ide, net in enumerate(nets):
            if metric == 'nll':
                # Forward pass
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
                    
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
                    
            else:
                # Forward pass
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    fut_pred_max = torch.zeros_like(fut_pred[0])
                    for k in range(lat_pred.shape[0]):
                        lat_man = torch.argmax(lat_pred[k, :]).detach()
                        lon_man = torch.argmax(lon_pred[k, :]).detach()
                        indx = lon_man*3 + lat_man
                        fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
                    l, c = maskedMSETest(fut_pred_max, fut, op_mask)
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedMSETest(fut_pred, fut, op_mask)
            for b in range(batch_size):
                fut_pred[:, b, 0] = fut_pred[:, b, 0] + fut_refPos[b, 0]
                fut_pred[:, b, 1] = fut_pred[:, b, 1] + fut_refPos[b, 1]
            deep_ensemble[ide, :, : ,:] = fut_pred

    
        ### For a specific vehicle
        if i >= 84:
            for select in range(batch_size):
                print(i, select)
                # muX = (fut_pred[:, select, 0]+fut_refPos[select, 0]).detach().cpu().numpy()*0.3048
                # muY = (fut_pred[:, select, 1]+fut_refPos[select, 1]).detach().cpu().numpy()*0.3048
                hx = (hist[:, select, 0]+hist_refPos[select, 0]).detach().cpu().numpy()*0.3048
                hy = (hist[:, select, 1]+hist_refPos[select, 1]).detach().cpu().numpy()*0.3048
                x = (fut[:, select, 0]+fut_refPos[select, 0]).detach().cpu().numpy()*0.3048
                y = (fut[:, select, 1]+fut_refPos[select, 1]).detach().cpu().numpy()*0.3048
                
                
                ensemble_x = deep_ensemble[:, :, select, 0]
                ensemble_y = deep_ensemble[:, :, select, 1]
                
                # uncertainty
                muX, muY, uncertainty_vx, uncertainty_cv, uncertainty_cv, uncertainty_vy = caluate_uncertainty(deep_ensemble, select)
                ensemble_plot(ensemble_x, ensemble_y, x, y, hx, hy)


            

        ###
                
        lossVals +=l.detach()
        counts += c.detach()


    if metric == 'nll':
        print(lossVals / counts)
    else:
        print(torch.pow(lossVals / counts,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters


