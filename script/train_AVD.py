import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import loss
from models import DeepMapping_AVD
from dataset_loader import AVD,AVDtrain

torch.backends.cudnn.deterministic = True
torch.manual_seed(2333)


parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-e','--n_epochs',type=int,default=1000,help='number of epochs')
parser.add_argument('-b','--batch_size',type=int,default=16,help='batch_size')
parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
parser.add_argument('-n','--n_samples',type=int,default=35,help='number of sampled unoccupied points along rays')
parser.add_argument('-s','--subsample_rate',type=int,default=40,help='subsample rate')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('-d','--data_dir',type=str,default='../data/ActiveVisionDataset/Home_011_1',help='dataset path')
parser.add_argument('-t','--traj',type=str,default='traj1.txt',help='trajectory file name')
parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
#parser.add_argument('-i','--init', type=str, default=None,help='init pose')
parser.add_argument('-y','--latent_size',type=int, default=16, help='length of latent')
parser.add_argument('-c','--conv_size',type=int, default=16, help='length of latent')
parser.add_argument('--log_interval',type=int,default=10,help='logging interval of saving results')
parser.add_argument('-o','--mode',type=str, default='simgle', help='length of latent')
parser.add_argument('--scene_index',type=int, default=None, help='scene index')
opt = parser.parse_args()
print(opt)
checkpoint_dir = os.path.join('../results/AVD',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO warm start
#if opt.init is not None:
#    init_pose_np = np.load(opt.init)
#    init_pose = torch.from_numpy(init_pose_np)
#else:
#    init_pose = None

latent_size = opt.latent_size
instances_per_scene = 16

latent_vecs = []
for i in range(instances_per_scene):
    vec = nn.Parameter(torch.ones(latent_size).normal_(0, 0.8).to(device))
    latent_vecs.append(vec)

print("latent size single:",latent_size)


print('loading dataset')
dataset = AVDtrain(opt.data_dir,opt.traj,instances_per_scene,opt.subsample_rate)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=False)
loss_fn = eval('loss.'+opt.loss)

print('creating model')

model = DeepMapping_AVD(latent_size,loss_fn=loss_fn,n_samples=opt.n_samples).to(device)
#model = nn.DataParallel(model,device_ids=[1,2]) 
optimizer = optim.Adam(
            [
                {
                     "params":model.parameters(), "lr":opt.lr,
                },
                {
                     "params": latent_vecs, "lr":opt.lr*4,  
                }
            ]
            )

print('start training')
best_loss = 2.0
pad_req_num = int(opt.conv_size/2)
for epoch in range(opt.n_epochs):
    training_loss= 0.0
    model.train()

    for index,(obs_batch,valid_pt,latent_indexes) in enumerate(loader):
        lats_inds = latent_indexes.cpu().detach().numpy() 
        latent_inputs = torch.ones((lats_inds.shape[0],opt.latent_size), dtype=torch.float32, requires_grad=True).to(device) 
        i = 0
        for i_lat in lats_inds:
            #latent_inputs = torch.cat([latent_inputs,latent_vecs[i_lat].unsqueeze(0)], 0)
            latent_inputs[i] *= latent_vecs[i_lat]
            i += 1
        
        
        #print(obs_batch.size(),valid_pt.size(),latent_indexes.size())
        latent_repeat = latent_inputs
        obs_batch = obs_batch.to(device)
        
        valid_pt = valid_pt.to(device)
        loss = model(latent_repeat,obs_batch,valid_pt)

        optimizer.zero_grad()
        loss.backward()
        #print(latent_vecs[lats_inds[0]])
        #print(latent_vecs[lats_inds[0]].grad)
        optimizer.step()
        #print(latent_vecs[lats_inds[0]])
        training_loss += loss.item()
    
    training_loss_epoch = training_loss/len(loader)

    if (epoch+1) % opt.log_interval == 0:
        print('[{}/{}], training loss: {:.4f}'.format(
            epoch+1,opt.n_epochs,training_loss_epoch))

    if  (epoch+1) % (4*opt.log_interval) == 0: 
        best_loss = training_loss_epoch
        obs_global_est_np = []
        pose_est_np = []
        with torch.no_grad():
            model.eval()
            for index,(obs_batch,valid_pt,latent_indexes) in enumerate(loader):
                lats_inds = latent_indexes.cpu().detach().numpy() 
                latent_inputs = torch.ones((lats_inds.shape[0],opt.latent_size), dtype=torch.float32, requires_grad=True).to(device) 
                i = 0
                for i_lat in lats_inds:
                    #latent_inputs = torch.cat([latent_inputs,latent_vecs[i_lat].unsqueeze(0)], 0)
                    latent_inputs[i] *= latent_vecs[i_lat]
                    i += 1
                latent_repeat = latent_inputs
                obs_batch = obs_batch.to(device)
                valid_pt = valid_pt.to(device)
                model(latent_repeat,obs_batch,valid_pt)

                obs_global_est_np.append(model.obs_global_est.cpu().detach().numpy())
                pose_est_np.append(model.pose_est.cpu().detach().numpy())
            
            pose_est_np = np.concatenate(pose_est_np)
            #if init_pose is not None:
            #    pose_est_np = utils.cat_pose_AVD(init_pose_np,pose_est_np)

            save_name = os.path.join(checkpoint_dir,'model_best.pth')
            utils.save_checkpoint(save_name,model,optimizer)

            obs_global_est_np = np.concatenate(obs_global_est_np)
            #kwargs = {'e':epoch+1}
            #valid_pt_np = dataset.valid_points.cpu().detach().numpy()

            save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
            np.save(save_name,obs_global_est_np)

            save_name = os.path.join(checkpoint_dir,'pose_est.npy')
            np.save(save_name,pose_est_np)
            os.system('./run_eval_vis_AVD.sh ' +str(opt.scene_index))
