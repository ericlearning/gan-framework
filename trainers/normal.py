import os
import cv2
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *


class Trainer():
    def __init__(self, loss_type, nz, netD, netG, device, train_dl, betas,
                 lr_D=0.0002, lr_G=0.0002, lr_decay='none', d_iter=5, g_iter=1,
                 loss_interval=50, image_interval=50, save_dir='saved/'):

        self.loss_type = loss_type
        self.netD = netD
        self.netG = netG
        self.device = device

        self.dl = train_dl
        self.train_dl = iter(self.dl)

        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr_D, betas)
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr_G, betas)
        self.lr_decay = lr_decay

        self.d_iter = d_iter
        self.g_iter = g_iter
        self.loss = get_gan_loss(self.device, loss_type)

        self.nz = nz
        self.fixed_z = self.generate_z(49, self.nz, self.device)
        self.loss_interval = loss_interval
        self.image_interval = image_interval

        self.errD_records = []
        self.errG_records = []

        self.save_cnt = 0
        self.save_dir = save_dir
        if(not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

    def generate_z(self, bs, nz, device):
        noise = torch.randn(bs, nz, device=device)
        return noise

    def disc_loss(self, xr, xf):
        self.netD.zero_grad()
        c_xr = self.netD(xr)
        c_xr = c_xr.view(-1)
        c_xf = self.netD(xf.detach())
        c_xf = c_xf.view(-1)
        errD = self.loss.d_loss(c_xr, c_xf)

        if self.loss_type in ['WGANGP', 'R1']:
            gp = self.loss.gradient_penalty(self.netD, xr, xf)
            errD = errD + gp * 10

        return errD

    def gen_loss(self, xr, xf):
        self.netG.zero_grad()
        c_xf = self.netD(xf)
        c_xf = c_xf.view(-1)
        errG = self.loss.g_loss(c_xf)
        return errG

    def progress_log(self, errD, errG, cur_iter, n_iter):
        if(cur_iter % self.loss_interval == 0):
            print('[%d/%d] errD : %.4f, errG : %.4f'
                  % (cur_iter + 1, n_iter, errD, errG))

        if(cur_iter % self.image_interval == 0):
            name = f'{self.save_cnt}:{cur_iter}.jpg'
            sample_images_list = get_sample_images_list(
                self.fixed_z, self.netG)
            plot_img = get_display_samples(
                sample_images_list, 7, 7)

            cur_file_name = os.path.join(self.save_dir, name)
            self.save_cnt += 1
            cv2.imwrite(cur_file_name, plot_img)

    def generate_sample(self, bs):
        z = self.generate_z(bs, self.nz, self.device)
        xf = self.netG(z)
        return xf

    def train_step(self, schedulers):
        schedulerD, schedulerG = schedulers
        for _ in range(self.g_iter):
            xr = self.get_real()
            xf = self.generate_sample(xr.shape[0])
            errG = self.gen_loss(xr, xf)
            errG.backward()
            self.optimizerG.step()
        schedulerG.step()

        for _ in range(self.d_iter):
            xr = self.get_real()
            xf = self.generate_sample(xr.shape[0])
            errD = self.disc_loss(xr, xf)
            errD.backward()
            self.optimizerD.step()
        schedulerD.step()

        self.errD_records.append(float(errD))
        self.errG_records.append(float(errG))
        return errD, errG

    def get_scheduler(self, total_iter):
        def schd(i):
            if(self.lr_decay == 'linear'):
                return (1 - i / total_iter)
            elif(self.lr_decay == 'none'):
                return 1.0
        schedulerD = LambdaLR(self.optimizerD, schd)
        schedulerG = LambdaLR(self.optimizerG, schd)
        return (schedulerD, schedulerG)

    def get_real(self):
        try:
            xr, _ = next(self.train_dl)
        except StopIteration:
            self.train_dl = iter(self.dl)
            xr, _ = next(self.train_dl)
        return xr.to(self.device)

    def train(self, n_iter):
        schedulers = self.get_scheduler(n_iter)
        for cur_iter in tqdm(range(n_iter)):
            errD, errG = self.train_step(schedulers)
            self.progress_log(errD, errG, cur_iter, n_iter)


class HAGANTrainer(Trainer):
    def __init__(self, object_num, **kwargs):
        super().__init__(**kwargs)
        self.object_num = object_num
        self.fixed_z = self.generate_noise(10, self.nz, self.device)

    def generate_noise(self, bs, nz, device):
        noise = torch.randn(self.object_num, bs, nz, device=device)
        return noise

    def generate_sample(self, bs):
        z = self.generate_noise(bs, self.nz, self.device)
        xf, _, _, _ = self.netG(z)
        return xf

    def progress_log(self, errD, errG, cur_iter, cur_epoch, n_iter, n_epoch):
        is_iter = cur_epoch is None and n_epoch is None
        if(cur_iter % self.loss_interval == 0):
            if is_iter:
                print('[%d/%d] errD : %.4f, errG : %.4f'
                      % (cur_iter + 1, n_iter, errD, errG))
            else:
                print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
                      % (cur_epoch + 1, n_epoch, cur_iter + 1,
                         n_iter, errD, errG))

        if(cur_iter % self.image_interval == 0):
            if is_iter:
                name = f'{self.save_cnt}:{cur_iter}.jpg'
            else:
                name = f'{self.save_cnt}:{cur_epoch}-{cur_iter}.jpg'
            sample_images_list = get_sample_images_list_hagan(
                self.fixed_z, self.netG)
            plot_img = get_display_samples(
                sample_images_list, 2 * 10, self.object_num + 1)

            cur_file_name = os.path.join(self.save_dir, name)
            self.save_cnt += 1
            cv2.imwrite(cur_file_name, plot_img)
