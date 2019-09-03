# -*- coding:utf-8 -*-


"""

@author: Jan
@file: test.py
@time: 2019/9/2 12:42
"""
## python-packages
import os
import torch
from prettytable import PrettyTable
import matplotlib.pyplot as plt
## self-defined packages
from config import parser
from data import UnAlignedDataLoader, tensor2image_RGB
from networks import Encoder, Decoder


def test(opt):
    #### mkdir
    des_pth = os.path.join('results', opt.name)
    if os.path.exists(os.path.join(des_pth)) is not True:
        os.mkdir(des_pth)
    src_pth = os.path.join(opt.checkpoints, opt.name)

    models_name = os.listdir(src_pth)
    models_name.remove('images')
    models_name.remove('records.txt')
    models_name.sort(key=lambda x: int(x[6:9]))
    target = int(models_name[-1][6:9])

    #### device
    device = torch.device('cuda:{}'.format(opt.gpu_id) if opt.gpu_id >= 0 else torch.device('cpu'))

    #### data
    data_loader = UnAlignedDataLoader()
    data_loader.initialize(opt)
    data_set = data_loader.load_data()

    #### networks
    ## initialize
    E_a2b = Encoder(input_nc=opt.input_nc, ngf=opt.ngf, norm_type=opt.norm_type, use_dropout=not opt.no_dropout, n_blocks=9)
    G_b = Decoder(output_nc=opt.output_nc, ngf=opt.ngf, norm_type=opt.norm_type)
    E_b2a = Encoder(input_nc=opt.input_nc, ngf=opt.ngf, norm_type=opt.norm_type, use_dropout=not opt.no_dropout, n_blocks=9)
    G_a = Decoder(output_nc=opt.output_nc, ngf=opt.ngf, norm_type=opt.norm_type)

    ## load in models
    E_a2b.load_state_dict(torch.load(os.path.join(src_pth, 'epoch_%3d-E_a2b.pth'%target)))
    G_b.load_state_dict(torch.load(os.path.join(src_pth, 'epoch_%3d-G_b.pth'%target)))
    E_b2a.load_state_dict(torch.load(os.path.join(src_pth, 'epoch_%3d-E_b2a.pth' % target)))
    G_a.load_state_dict(torch.load(os.path.join(src_pth, 'epoch_%3d-G_a.pth' % target)))

    E_a2b = E_a2b.to(device)
    G_b = G_b.to(device)
    E_b2a = E_b2a.to(device)
    G_a = G_a.to(device)

    for i, data in enumerate(data_set):
        real_A = data['A'].to(device)
        real_B = data['B'].to(device)

        fake_B = G_b(E_a2b(real_A))
        fake_A = G_a(E_b2a(real_B))

        ## visualize
        if opt.gpu_id >= 0:
            fake_B = fake_B.cpu().data
            fake_A = fake_A.cpu().data

            real_A = real_A.cpu()
            real_B = real_B.cpu()

        for j in range(opt.batch_size):
            fake_b = tensor2image_RGB(fake_B[j, ...])
            fake_a = tensor2image_RGB(fake_A[j, ...])

            real_a = tensor2image_RGB(real_A[j, ...])
            real_b = tensor2image_RGB(real_B[j, ...])

            plt.subplot(221), plt.title("real_A"), plt.imshow(real_a)
            plt.subplot(222), plt.title("fake_B"), plt.imshow(fake_b)
            plt.subplot(223), plt.title("real_B"), plt.imshow(real_b)
            plt.subplot(224), plt.title("fake_A"), plt.imshow(fake_a)

            plt.savefig(os.path.join(des_pth, '%06d-%02d.jpg'%(i, j)))
        #break #-> debug

    print("≧◔◡◔≦ Congratulation! Successfully finishing the testing!")


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.mode = 'test'

    #### opt setting
    table = PrettyTable(field_names=['config-name', 'config-value'])
    table.align['config-name'] = 'm'
    table.align['config-value'] = 'm'

    for k, v in sorted(vars(opt).items()):
        table.add_row([k, v])
    print(table.get_string(reversesort=True))

    #### testing
    test(opt)
