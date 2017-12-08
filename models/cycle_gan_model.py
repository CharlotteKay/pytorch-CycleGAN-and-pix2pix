import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.A_mask = self.Tensor(nb, opt.input_nc, size, size)
        self.B_mask = self.Tensor(nb, opt.input_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A1 = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D1, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_A2 = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D2, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B1 = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D1, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B2 = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D2, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A1, 'D_A1', which_epoch)
                self.load_network(self.netD_A2, 'D_A2', which_epoch)
                self.load_network(self.netD_B1, 'D_B1', which_epoch)
                self.load_network(self.netD_B2, 'D_B2', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A1 = torch.optim.Adam(self.netD_A1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A2 = torch.optim.Adam(self.netD_A2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B1 = torch.optim.Adam(self.netD_B1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B2 = torch.optim.Adam(self.netD_B2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A1)
            self.optimizers.append(self.optimizer_D_A2)
            self.optimizers.append(self.optimizer_D_B1)
            self.optimizers.append(self.optimizer_D_B2)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A1)
            networks.print_network(self.netD_A2)
            networks.print_network(self.netD_B1)
            networks.print_network(self.netD_B2)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.face_mask:
            A_mask = input['A_mask']
            B_mask = input['B_mask']
            self.A_mask = Variable(A_mask, requires_grad=False).cuda()
            self.B_mask = Variable(B_mask, requires_grad=False).cuda()

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A1(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A1 = self.backward_D_basic(self.netD_A1, self.real_B, fake_B) 

    def backward_D_A2(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A2 = self.backward_D_basic(self.netD_A2, self.real_B, fake_B) 

    def backward_D_B1(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B1 = self.backward_D_basic(self.netD_B1, self.real_A, fake_A)

    def backward_D_B2(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B2 = self.backward_D_basic(self.netD_B2, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake1 = self.netD_A1.forward(self.fake_B)
        pred_fake2 = self.netD_A2.forward(self.fake_B)
        self.loss_G_A = (self.criterionGAN(pred_fake1, True) + self.criterionGAN(pred_fake2, True)) * 0.5
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake1 = self.netD_B1.forward(self.fake_A)
        pred_fake2 = self.netD_B2.forward(self.fake_A)
        self.loss_G_B = (self.criterionGAN(pred_fake1, True) + self.criterionGAN(pred_fake2, True)) * 0.5
        
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        if not self.opt.face_mask:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        else:
            self.loss_cycle_A = ((self.rec_A - self.real_A).abs() * self.A_mask).mean() * lambda_A
        
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        if not self.opt.face_mask:
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        else:
            self.loss_cycle_B = ((self.rec_B - self.real_B).abs() * self.B_mask).mean() * lambda_B

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A1
        self.optimizer_D_A1.zero_grad()
        self.backward_D_A1()
        self.optimizer_D_A1.step()
        # D_A2
        self.optimizer_D_A2.zero_grad()
        self.backward_D_A2()
        self.optimizer_D_A2.step()
        # D_B1
        self.optimizer_D_B1.zero_grad()
        self.backward_D_B1()
        self.optimizer_D_B1.step()
        # D_B2
        self.optimizer_D_B2.zero_grad()
        self.backward_D_B2()
        self.optimizer_D_B2.step()

    def get_current_errors(self):
        D_A1 = self.loss_D_A1.data[0]
        D_A2 = self.loss_D_A2.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        D_B1 = self.loss_D_B1.data[0]
        D_B2 = self.loss_D_B2.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A1', D_A1), ('D_A2', D_A2), ('G_A', G_A), ('Cyc_A', Cyc_A), 
                                ('idt_A', idt_A), ('D_B1', D_B1), ('D_B2', D_B2), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A1', D_A1), ('D_A2', D_A2), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B1', D_B1), ('D_B2', D_B2), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.face_mask:
            mask_A = util.mask2im(self.A_mask.data, face_weight=self.opt.face_weight)
            mask_B = util.mask2im(self.B_mask.data, face_weight=self.opt.face_weight)
        if self.opt.isTrain and self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        elif self.opt.face_mask:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                ('mask_A', mask_A), ('mask_B', mask_B)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A1, 'D_A1', label, self.gpu_ids)
        self.save_network(self.netD_A2, 'D_A2', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B1, 'D_B1', label, self.gpu_ids)
        self.save_network(self.netD_B2, 'D_B2', label, self.gpu_ids)
