import os
import torch
import torch.nn as nn
from dataset import Dataset
from trainers.normal import Trainer
from utils import save, load
from architectures.models.sngan import SNDCGANDiscriminator, SNDCGANGenerator

torch.backends.cudnn.benchmark = True

dir_name = 'data/celeba'
basic_types = None

sz, nz = 64, 128
lr_D, lr_G, betas, bs = 0.0002, 0.0002, (0, 0.99), 64
save_dir = 'saved/'

trn_ds = Dataset(dir_name, basic_types)
trn_dl = trn_ds.get_loader(sz, bs, num_workers=10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = SNDCGANDiscriminator(64).to(device)
netG = SNDCGANGenerator(64, nz).to(device)

trainer = Trainer('R1', nz, netD, netG, device, trn_dl, betas, lr_D, lr_G,
                  d_iter=1, g_iter=1, loss_interval=150, image_interval=300,
                  save_dir=save_dir)

trainer.train(100000)
save('saved/cur_state.state', netD, netG,
     trainer.optimizerD, trainer.optimizerG)
torch.save(netG.state_dict(), 'saved/cur_state_G.pth')
