import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

from models import weights_init, weights_init_text, Discriminator, Generator, DiscriminatorText
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
import lpips

from torch.autograd import Variable

policy = 'color,translation'
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True if torch.cuda.is_available() else False)


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def compute_generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fake_features = netD(fake_imgs)
    
    inputs = (fake_features, cond)
    fake_logits = netD.get_cond_logits(inputs)
    errD_fake = criterion(fake_logits, real_labels)

    if netD.get_uncond_logits is not None:
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    
    return errD_fake

def compute_text_discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, conditions):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs)

    # real pairs
    inputs = (real_features, cond)
    real_logits = netD.get_cond_logits(real_features, cond)
    errD_real = criterion(real_logits, real_labels)

    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = netD.get_cond_logits(real_features[:(batch_size-1)], cond[1:])
    errD_wrong = criterion(wrong_logits, fake_labels[1:])

    errD = errD_real + errD_wrong

    return errD, errD_real.item(), errD_wrong.item()

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        feat, pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        return err, feat
        #return  err, feat_16, pred.mean().item(), rec_all, rec_small, rec_part
    else:
        feat, pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()

        return err, feat
        

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = args.start_iter
    save_interval = 100
    kl_cf = args.kl
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_list = [
        transforms.Resize((int(im_size),int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

   
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netDText = DiscriminatorText()
    netDText.apply(weights_init_text)

    netG.to(device)
    netD.to(device)
    netDText.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    real_labels = Variable(torch.FloatTensor(args.batch_size).fill_(1)).to(device)
    fake_labels = Variable(torch.FloatTensor(args.batch_size).fill_(0)).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerDText = optim.Adam(netDText.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image, txt_embedding, txt = next(dataloader)

        real_image = real_image.to(device)
        txt_embedding = txt_embedding.to(device)
        
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images, mu, logvar = netG(noise, txt_embedding)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()

        errDReal, feat_real = train_d(netD, real_image, label="real")
        errDFake, feat_fake = train_d(netD, [fi.detach() for fi in fake_images], label="fake")

        errDReal.backward()
        errDFake.backward()
        optimizerD.step()
        
        ## 3. train text Discriminator
        netD.zero_grad()
        netDText.zero_grad()

        errD, _, _ = compute_text_discriminator_loss(netDText, feat_real.detach(), feat_fake.detach(), real_labels, fake_labels, mu)
        errD.backward()

        optimizerDText.step()

        ## 4. train Generator
        netG.zero_grad()
        _, pred_g = netD(fake_images, "fake") + compute_generator_loss(netDText, feat_fake.detach(), real_labels, mu)
        err_g = -pred_g.mean()
        
        kl_loss = KL_loss(mu, logvar)
        errG_total = err_g + kl_loss * kl_cf

        errG_total.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        # if iteration % 100 == 0:
        #     print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -errG_total.item()))
          
        # if iteration % (save_interval*10) == 0:
        #     backup_para = copy_G_params(netG)
        #     load_params(netG, avg_param_G)
        #     with torch.no_grad():
        #         vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
        #         vutils.save_image( torch.cat([
        #                 F.interpolate(real_image, 128), 
        #                 rec_img_all, rec_img_small,
        #                 rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
        #     load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='./dataset', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--t-dim', type=int, default=768)
    parser.add_argument('--c-dim', type=int, default=128)
    parser.add_argument('--kl', type=float, default=2.0)

    args = parser.parse_args()
    print(args)

    train(args)
