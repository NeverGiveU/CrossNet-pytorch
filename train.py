# -*- coding:utf-8 -*-


"""

@author: Jan
@file: train.py
@time: 2019/9/1 13:22
"""
## python-packages
import time
import os
from prettytable import PrettyTable
import torch.nn as nn
import torch
import itertools
import matplotlib.pyplot as plt
## self-defined packages
from config import parser
from data import UnAlignedDataLoader, ImagePool
from data import tensor2image_RGB
from networks import Encoder, Decoder, LatentTranslator, Discriminator
from networks import init_net, set_requires_grad, get_scheduler, update_learning_rate
from loss import GANLoss


def train(opt):
    #### device
    device = torch.device('cuda:{}'.format(opt.gpu_id) if opt.gpu_id>=0 else torch.device('cpu'))

    #### dataset
    data_loader = UnAlignedDataLoader()
    data_loader.initialize(opt)
    data_set = data_loader.load_data()
    print("The number of training images = %d." % len(data_set))

    #### initialize models
    ## declaration
    E_a2Zb = Encoder(input_nc=opt.input_nc, ngf=opt.ngf, norm_type=opt.norm_type, use_dropout=not opt.no_dropout, n_blocks=9)
    G_Zb2b = Decoder(output_nc=opt.output_nc, ngf=opt.ngf, norm_type=opt.norm_type)
    T_Zb2Za = LatentTranslator(n_channels=256, norm_type=opt.norm_type, use_dropout=not opt.no_dropout)
    D_b = Discriminator(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=opt.n_layers, norm_type=opt.norm_type)

    E_b2Za = Encoder(input_nc=opt.input_nc, ngf=opt.ngf, norm_type=opt.norm_type, use_dropout=not opt.no_dropout, n_blocks=9)
    G_Za2a = Decoder(output_nc=opt.output_nc, ngf=opt.ngf, norm_type=opt.norm_type)
    T_Za2Zb = LatentTranslator(n_channels=256, norm_type=opt.norm_type, use_dropout=not opt.no_dropout)
    D_a = Discriminator(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=opt.n_layers, norm_type=opt.norm_type)

    ## initialization
    E_a2Zb = init_net(E_a2Zb, init_type=opt.init_type).to(device)
    G_Zb2b = init_net(G_Zb2b, init_type=opt.init_type).to(device)
    T_Zb2Za = init_net(T_Zb2Za, init_type=opt.init_type).to(device)
    D_b = init_net(D_b, init_type=opt.init_type).to(device)

    E_b2Za = init_net(E_b2Za, init_type=opt.init_type).to(device)
    G_Za2a = init_net(G_Za2a, init_type=opt.init_type).to(device)
    T_Za2Zb = init_net(T_Za2Zb, init_type=opt.init_type).to(device)
    D_a = init_net(D_a, init_type=opt.init_type).to(device)
    print("+------------------------------------------------------+\nFinish initializing networks.")

    #### optimizer and criterion
    ## criterion
    criterionGAN = GANLoss(opt.gan_mode).to(device)
    criterionZId = nn.L1Loss()
    criterionIdt = nn.L1Loss()
    criterionCTC = nn.L1Loss()
    criterionZCyc= nn.L1Loss()

    ## optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(E_a2Zb.parameters(), G_Zb2b.parameters(), T_Zb2Za.parameters(),
                                                   E_b2Za.parameters(), G_Za2a.parameters(), T_Za2Zb.parameters()),
                                   lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_D = torch.optim.Adam(itertools.chain(D_a.parameters(), D_b.parameters()),
                                   lr=opt.lr, betas=(opt.beta1, opt.beta2))

    ## scheduler
    scheduler = [get_scheduler(optimizer_G, opt), get_scheduler(optimizer_D, opt)]

    print("+------------------------------------------------------+\nFinish initializing the optimizers and criterions.")

    #### global variables
    checkpoints_pth = os.path.join(opt.checkpoints, opt.name)
    if os.path.exists(checkpoints_pth) is not True:
        os.mkdir(checkpoints_pth)
        os.mkdir(os.path.join(checkpoints_pth, 'images'))
    record_fh = open(os.path.join(checkpoints_pth, 'records.txt'), 'w', encoding='utf-8')
    loss_names = ['GAN_A', 'Adv_A', 'Idt_A', 'CTC_A', 'ZId_A', 'ZCyc_A', 'GAN_B', 'Adv_B', 'Idt_B', 'CTC_B', 'ZId_B', 'ZCyc_B']

    fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
    fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

    print("+------------------------------------------------------+\nFinish preparing the other works.")
    print("+------------------------------------------------------+\nNow training is beginning ..")
    #### training
    cur_iter = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch

        for i, data in enumerate(data_set):
            ## setup inputs
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)

            ## forward
            # image cycle / GAN
            latent_B = E_a2Zb(real_A)               #-> a -> Zb     : E_a2b(a)
            fake_B = G_Zb2b(latent_B)               #-> Zb -> b'    : G_b(E_a2b(a))
            latent_A = E_b2Za(real_B)               #-> b -> Za     : E_b2a(b)
            fake_A = G_Za2a(latent_A)               #-> Za -> a'    : G_a(E_b2a(b))

            # Idt
            '''
            rec_A = G_Za2a(E_b2Za(fake_B))          #-> b' -> Za' -> rec_a  : G_a(E_b2a(fake_b))
            rec_B = G_Zb2b(E_a2Zb(fake_A))          #-> a' -> Zb' -> rec_b  : G_b(E_a2b(fake_a))
            '''
            idt_latent_A = E_b2Za(real_A)           #-> a -> Za        : E_b2a(a)
            idt_A = G_Za2a(idt_latent_A)            #-> Za -> idt_a    : G_a(E_b2a(a))
            idt_latent_B = E_a2Zb(real_B)           #-> b -> Zb        : E_a2b(b)
            idt_B = G_Zb2b(idt_latent_B)            #-> Zb -> idt_b    : G_b(E_a2b(b))

            # ZIdt
            T_latent_A = T_Zb2Za(latent_B)          #-> Zb -> Za''  : T_b2a(E_a2b(a))
            T_rec_A = G_Za2a(T_latent_A)            #-> Za'' -> a'' : G_a(T_b2a(E_a2b(a)))
            T_latent_B = T_Za2Zb(latent_A)          #-> Za -> Zb''  : T_a2b(E_b2a(b))
            T_rec_B = G_Zb2b(T_latent_B)            #-> Zb'' -> b'' : G_b(T_a2b(E_b2a(b)))

            # CTC
            T_idt_latent_B = T_Za2Zb(idt_latent_A)  #-> a -> T_a2b(E_b2a(a))
            T_idt_latent_A = T_Zb2Za(idt_latent_B)  #-> b -> T_b2a(E_a2b(b))

            # ZCyc
            TT_latent_B = T_Za2Zb(T_latent_A)       #-> T_a2b(T_b2a(E_a2b(a)))
            TT_latent_A = T_Zb2Za(T_latent_B)       #-> T_b2a(T_a2b(E_b2a(b)))

            ### optimize parameters
            ## Generator updating
            set_requires_grad([D_b, D_a], False) #-> set Discriminator to require no gradient
            optimizer_G.zero_grad()
            # GAN loss
            loss_G_A = criterionGAN(D_b(fake_B), True)
            loss_G_B = criterionGAN(D_a(fake_A), True)
            loss_GAN = loss_G_A + loss_G_B
            # Idt loss
            loss_idt_A = criterionIdt(idt_A, real_A)
            loss_idt_B = criterionIdt(idt_B, real_B)
            loss_Idt = loss_idt_A + loss_idt_B
            # Latent cross-identity loss
            loss_Zid_A = criterionZId(T_rec_A, real_A)
            loss_Zid_B = criterionZId(T_rec_B, real_B)
            loss_Zid = loss_Zid_A + loss_Zid_B
            # Latent cross-translation consistency
            loss_CTC_A = criterionCTC(T_idt_latent_A, latent_A)
            loss_CTC_B = criterionCTC(T_idt_latent_B, latent_B)
            loss_CTC = loss_CTC_B + loss_CTC_A
            # Latent cycle consistency
            loss_ZCyc_A = criterionZCyc(TT_latent_A, latent_A)
            loss_ZCyc_B = criterionZCyc(TT_latent_B, latent_B)
            loss_ZCyc= loss_ZCyc_B + loss_ZCyc_A

            loss_G = opt.lambda_gan * loss_GAN + opt.lambda_idt * loss_Idt + opt.lambda_zid * loss_Zid + opt.lambda_ctc * loss_CTC + opt.lambda_zcyc * loss_ZCyc

            # backward and gradient updating
            loss_G.backward()
            optimizer_G.step()

            ## Discriminator updating
            set_requires_grad([D_b, D_a], True)  # -> set Discriminator to require gradient
            optimizer_D.zero_grad()

            # backward D_b
            fake_B_ = fake_B_pool.query(fake_B)
            #-> real_B, fake_B
            pred_real_B = D_b(real_B)
            loss_D_real_B = criterionGAN(pred_real_B, True)

            pred_fake_B = D_b(fake_B_)
            loss_D_fake_B = criterionGAN(pred_fake_B, False)

            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            loss_D_B.backward()

            # backward D_a
            fake_A_ = fake_A_pool.query(fake_A)
            #-> real_A, fake_A
            pred_real_A = D_a(real_A)
            loss_D_real_A = criterionGAN(pred_real_A, True)

            pred_fake_A = D_a(fake_A_)
            loss_D_fake_A = criterionGAN(pred_fake_A, False)

            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            loss_D_A.backward()

            # update the gradients
            optimizer_D.step()

            ### validate here, both qualitively and quantitatively
            ## record the losses
            if cur_iter % opt.log_freq == 0:
                # loss_names = ['GAN_A', 'Adv_A', 'Idt_A', 'CTC_A', 'ZId_A', 'ZCyc_A', 'GAN_B', 'Adv_B', 'Idt_B', 'CTC_B', 'ZId_B', 'ZCyc_B']
                losses = [loss_G_A.item(), loss_D_A.item(), loss_idt_A.item(), loss_CTC_A.item(), loss_Zid_A.item(), loss_ZCyc_A.item(),
                          loss_G_B.item(), loss_D_B.item(), loss_idt_B.item(), loss_CTC_B.item(), loss_Zid_B.item(), loss_ZCyc_B.item()]
                # record
                line = ''
                for loss in losses:
                    line += '{} '.format(loss)
                record_fh.write(line[:-1] + '\n')
                # print out
                print('Epoch: %3d/%3dIter: %9d--------------------------+'%(epoch, opt.epoch, i))
                field_names = loss_names[: len(loss_names)//2]
                table = PrettyTable(field_names=field_names)
                for l_n in field_names:
                    table.align[l_n] = 'm'
                table.add_row(losses[: len(field_names)])
                print(table.get_string(reversesort=True))

                field_names = loss_names[len(loss_names)//2 : ]
                table = PrettyTable(field_names=field_names)
                for l_n in field_names:
                    table.align[l_n] = 'm'
                table.add_row(losses[-len(field_names) :])
                print(table.get_string(reversesort=True))

            ## visualize
            if cur_iter % opt.vis_freq == 0:
                if opt.gpu_id >= 0:
                    real_A =  real_A.cpu().data
                    real_B =  real_B.cpu().data
                    fake_A =  fake_A.cpu().data
                    fake_B =  fake_B.cpu().data
                    idt_A  =   idt_A.cpu().data
                    idt_B  =   idt_B.cpu().data
                    T_rec_A= T_rec_A.cpu().data
                    T_rec_B= T_rec_B.cpu().data

                plt.subplot(241), plt.title('real_A' ), plt.imshow(tensor2image_RGB( real_A[0, ...]))
                plt.subplot(242), plt.title('fake_B' ), plt.imshow(tensor2image_RGB( fake_B[0, ...]))
                plt.subplot(243), plt.title('idt_A'  ), plt.imshow(tensor2image_RGB(  idt_A[0, ...]))
                plt.subplot(244), plt.title('L_idt_A'), plt.imshow(tensor2image_RGB(T_rec_A[0, ...]))

                plt.subplot(245), plt.title('real_B' ), plt.imshow(tensor2image_RGB( real_B[0, ...]))
                plt.subplot(246), plt.title('fake_A' ), plt.imshow(tensor2image_RGB( fake_A[0, ...]))
                plt.subplot(247), plt.title('idt_B'  ), plt.imshow(tensor2image_RGB(  idt_B[0, ...]))
                plt.subplot(248), plt.title('L_idt_B'), plt.imshow(tensor2image_RGB(T_rec_B[0, ...]))

                plt.savefig(os.path.join(checkpoints_pth, 'images', '%03d_%09d.jpg'%(epoch, i)))

            cur_iter += 1
            #break #-> debug

        ## till now, we finish one epoch, try to update the learning rate
        update_learning_rate(schedulers=scheduler, opt=opt, optimizer=optimizer_D)
        ## save the model
        if epoch % opt.ckp_freq == 0:
            #-> save models
            # torch.save(model.state_dict(), PATH)
            #-> load in models
            # model.load_state_dict(torch.load(PATH))
            # model.eval()
            if opt.gpu_id >= 0:
                E_a2Zb = E_a2Zb.cpu()
                G_Zb2b = G_Zb2b.cpu()
                T_Zb2Za = T_Zb2Za.cpu()
                D_b = D_b.cpu()

                E_b2Za = E_b2Za.cpu()
                G_Za2a = G_Za2a.cpu()
                T_Za2Zb = T_Za2Zb.cpu()
                D_a = D_a.cpu()
                '''
                torch.save( E_a2Zb.cpu().state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-E_a2b.pth' % epoch))
                torch.save( G_Zb2b.cpu().state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-G_b.pth' % epoch))
                torch.save(T_Zb2Za.cpu().state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-T_b2a.pth' % epoch))
                torch.save(    D_b.cpu().state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-D_b.pth' % epoch))

                torch.save( E_b2Za.cpu().state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-E_b2a.pth' % epoch))
                torch.save( G_Za2a.cpu().state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-G_a.pth' % epoch))
                torch.save(T_Za2Zb.cpu().state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-T_a2b.pth' % epoch))
                torch.save(    D_a.cpu().state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-D_a.pth' % epoch))
                '''
            torch.save( E_a2Zb.state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-E_a2b.pth' % epoch))
            torch.save( G_Zb2b.state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-G_b.pth' % epoch))
            torch.save(T_Zb2Za.state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-T_b2a.pth' % epoch))
            torch.save(    D_b.state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-D_b.pth' % epoch))

            torch.save( E_b2Za.state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-E_b2a.pth' % epoch))
            torch.save( G_Za2a.state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-G_a.pth' % epoch))
            torch.save(T_Za2Zb.state_dict(), os.path.join(checkpoints_pth, 'epoch_%3d-T_a2b.pth' % epoch))
            torch.save(    D_a.state_dict(), os.path.join(checkpoints_pth,   'epoch_%3d-D_a.pth' % epoch))
            if opt.gpu_id >= 0:
                E_a2Zb = E_a2Zb.to(device)
                G_Zb2b = G_Zb2b.to(device)
                T_Zb2Za = T_Zb2Za.to(device)
                D_b = D_b.to(device)

                E_b2Za = E_b2Za.to(device)
                G_Za2a = G_Za2a.to(device)
                T_Za2Zb = T_Za2Zb.to(device)
                D_a = D_a.to(device)
            print("+Successfully saving models in epoch: %3d.-------------+" % epoch)
        #break #-> debug
    record_fh.close()
    print("≧◔◡◔≦ Congratulation! Finishing the training!")


if __name__ == '__main__':
    opt = parser.parse_args()

    #### opt setting
    table = PrettyTable(field_names=['config-name', 'config-value'])
    table.align['config-name'] = 'm'
    table.align['config-value'] = 'm'

    for k, v in sorted(vars(opt).items()):
        table.add_row([k, v])
    print(table.get_string(reversesort=True))

    #### training
    train(opt)
