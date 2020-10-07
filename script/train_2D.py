import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import loss
from models import DeepMapping2D
from dataset_loader import SimulatedPointCloud

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-e','--n_epochs',type=int,default=1000,help='number of epochs')
parser.add_argument('-b','--batch_size',type=int,default=32,help='batch_size')
parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
parser.add_argument('-n','--n_samples',type=int,default=19,help='number of sampled unoccupied points along rays')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
parser.add_argument('-i','--init', type=str, default=None,help='init pose')
parser.add_argument('--log_interval',type=int,default=10,help='logging interval of saving results')
parser.add_argument('-s','--seed',type=int,help='random start seed')
parser.add_argument('-k','--conv_size',type=int, help='convsize of latent vector') 
parser.add_argument('-y','--latent_size',type=int, help='latent size of latent vector')
parser.add_argument('-o','--mode',type=str,default='maxpool',help='mode of latent gen')
opt = parser.parse_args()
torch.manual_seed(opt.seed)
print('maxpool')
print("SEED",opt.seed)
checkpoint_dir = os.path.join('../results/2D',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


print('loading dataset')
if opt.init is not None:
    init_pose_np = np.load(opt.init)
    init_pose = torch.from_numpy(init_pose_np)
else:
    init_pose = None
    
latent_vecs = []
 
conv_size = opt.conv_size


latent_size = opt.latent_size
print("latent_size",latent_size)
instances_per_scene = 256
if opt.mode == 'single':
    print ('single')
    latent_vecs = torch.ones(latent_size).normal_(0, 0.8).to(device)
else:
    print("conv_size",conv_size)
    for i in range(instances_per_scene):
        vec = (torch.ones(latent_size).normal_(0, 0.8).to(device))
        vec.requires_grad = True
        latent_vecs.append(vec)

if conv_size>0 and opt.mode=="concat":
    latent_size = conv_size*latent_size

dataset = SimulatedPointCloud(opt.data_dir,instances_per_scene,init_pose)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=False)
#print(len(dataset))
loss_fn = eval('loss.'+opt.loss)

def maxpool_1D(latents):
    maxlat = torch.max(latents,0)
    #print(maxlat[0].size())
    return maxlat[0]

print('creating model')
model = DeepMapping2D(loss_fn=loss_fn,n_obs=dataset.n_obs, latent_size = latent_size, n_samples=opt.n_samples).to(device)
optimizer = optim.Adam(
            [
                {
                     "params":model.parameters(), "lr":opt.lr,
                },
                {
                     "params": latent_vecs, "lr":opt.lr,  
                }
            ]
            )

if opt.model is not None:
    utils.load_checkpoint(opt.model,model,optimizer)

print('start training')
for epoch in range(opt.n_epochs):

    training_loss= 0.0
    model.train()
    for index,(obs_batch,valid_pt,index_latents) in enumerate(loader):
        obs_batch = obs_batch.to(device)
        valid_pt = valid_pt.to(device)
        latent_inputs = torch.zeros(0).cuda()
        for i_lat in index_latents.cpu().detach().numpy():
            pad_req_num = int(conv_size/2)
            if opt.mode=='single':
                latent = latent_vecs
            elif opt.mode=='maxpool_single':
                latent = maxpool_1D(torch.stack(latent_vecs))
            else:
                start = i_lat-pad_req_num+1
                end = i_lat+pad_req_num
                if start<0:
                    latent = [latent_vecs[0] if i_c<0 else latent_vecs[i_c] for i_c in range(start,end+1)]
                    #latent = torch.cat([latent_vecs[i_lat],latent_vecs[i_lat]],0)
                elif end>instances_per_scene-1:
                    latent = [latent_vecs[i_c] if i_c<=instances_per_scene-1 else latent_vecs[-1] for i_c in range(start,end+1)]
                else: 
                    latent = [latent_vecs[i_c] for i_c in range(start,end+1)]
                latent = torch.stack(latent)
                latent = maxpool_1D(latent)
            #latent_repeat = latent.expand(obs_batch.shape[-2], -1).unsqueeze(0)
            latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(1)], 1)
        #latent_inputs=latent_inputs.permute(0,2,1)
        latent_inputs = latent_inputs.transpose(0,1) 
        obs_batch = obs_batch.to(device)
        loss = model(obs_batch,valid_pt,latent_inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    
    training_loss_epoch = training_loss/len(loader)
    
    if (epoch+1) % opt.log_interval == 0:
        print('[{}/{}], training loss: {:.4f}'.format(
            epoch+1,opt.n_epochs,training_loss_epoch))

        obs_global_est_np = []
        pose_est_np = []
        with torch.no_grad():
            model.eval()
            for index,(obs_batch,valid_pt,index_latents) in enumerate(loader):
                latent_inputs = torch.zeros(0).cuda()
                for i_lat in index_latents.cpu().detach().numpy():
                    pad_req_num = int(conv_size/2)
                    if opt.mode == 'single':
                        latent = latent_vecs
                    elif opt.mode == 'maxpool_single':
                        latent = maxpool_1D(torch.stack(latent_vecs))
                    elif conv_size<=1 and opt.mode=="concat":
                        latent = latent_vecs[i_lat]
                    else:
                        start = i_lat-pad_req_num+1
                        end = i_lat+pad_req_num
                        if start<0:
                            latent = [latent_vecs[0] if i_c<0 else latent_vecs[i_c] for i_c in range(start,end+1)]
                            #latent = torch.cat([latent_vecs[i_lat],latent_vecs[i_lat]],0)
                        elif end>instances_per_scene-1:
                            latent = [latent_vecs[i_c] if i_c<=instances_per_scene-1 else latent_vecs[-1] for i_c in range(start,end+1)]
                        else:
                            latent = [latent_vecs[i_c] for i_c in range(start,end+1)]
                        latent = torch.stack(latent)
                        latent = maxpool_1D(latent)
                    latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(1)], 1)
                latent_inputs = latent_inputs.transpose(0,1)
                #print("sdsd",latent_inputs)
                obs_batch = obs_batch.to(device)
                valid_pt = valid_pt.to(device)
                model(obs_batch,valid_pt,latent_inputs)
                obs_global_est_np.append(model.obs_global_est.cpu().detach().numpy())
                pose_est_np.append(model.pose_est.cpu().detach().numpy())
            
            pose_est_np = np.concatenate(pose_est_np)
            if init_pose is not None:
                pose_est_np = utils.cat_pose_2D(init_pose_np,pose_est_np)

            save_name = os.path.join(checkpoint_dir,'model_best.pth')
            utils.save_checkpoint(save_name,model,optimizer)
            z_ts = torch.stack(latent_vecs)
            torch.save(z_ts, os.path.join(checkpoint_dir,'model_best_z.pth'))
            obs_global_est_np = np.concatenate(obs_global_est_np)
            kwargs = {'e':epoch+1}
            valid_pt_np = dataset.valid_points.cpu().detach().numpy()
            utils.plot_global_point_cloud(obs_global_est_np,pose_est_np,valid_pt_np,checkpoint_dir,**kwargs)
            
            #save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
            #np.save(save_name,obs_global_est_np)

            save_name = os.path.join(checkpoint_dir,'pose_est.npy')
            np.save(save_name,pose_est_np)
            
    if (epoch+1) % (opt.log_interval*4) == 0:
        os.system('./run_eval_2D.sh')
