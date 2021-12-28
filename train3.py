"""
InfoGan training
"""
import torch
from torch import nn
import os
import cv2
import numpy
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
from train1 import tiny_imagenet
from infogan import *
from util import to_categorical


def main():
    iters = 0
    epochs = 50
    n_classes = 200
    code_dim = 20
    noise_dim = 0
    snapshot_path = 'infogan.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_list = []
    G_losses = []
    D_losses = []
    dataset = tiny_imagenet()
    batch_size = 20
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Create the netG
    netG = Generator(noise_dim, n_classes, code_dim).to(device)
    netD = Discriminator(n_classes, code_dim).to(device)
    if os.path.exists(snapshot_path):
        state_dict = torch.load(snapshot_path)
        netG.load_state_dict(state_dict['netG'])
        netD.load_state_dict(state_dict['netD'])

    netG.to(device)
    netD.to(device)

    netD_params = list(netD.parameters())
    netG_params = list(netG.parameters())
    lr = 0.0001
    beta1 = 0.5
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the netG
    z = torch.normal(0, 1, (batch_size, noise_dim))
    # Ground truth labels
    gt_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
    label_input = to_categorical(gt_labels, num_columns=n_classes).to(device)
    code_input = torch.as_tensor(numpy.random.uniform(-1, 1, (batch_size, code_dim))).to(device).float()
    fixed_noise = (z, label_input, code_input)

    adversarial_loss = nn.MSELoss()
    continuous_loss = nn.MSELoss()
    categorical_loss = nn.CrossEntropyLoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        print('STARTING EPOCH ', epoch + 1)
        for i, batch in enumerate(loader):
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            data, labels = batch
            b_size = len(data)
            ones = torch.full((b_size,), 1, dtype=torch.float, device=device)
            zeros = torch.full((b_size,), 0, dtype=torch.float, device=device)

            # Sample noise and labels as netG input
            z = torch.normal(0, 1, (b_size, noise_dim))
            # Ground truth labels
            gt_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            label_input = to_categorical(gt_labels, num_columns=n_classes).to(device)
            code_input = torch.as_tensor(numpy.random.uniform(-1, 1, (batch_size, code_dim))).to(device).float()

            # Generate a batch of images
            gen_imgs = netG(z, label_input, code_input)

            real_pred, real_label, real_code = netD(gen_imgs)
            # Loss measures netG's ability to fool the netD
            g_loss = adversarial_loss(real_pred, ones)
            fake_pred, fake_label, fake_code = netD(gen_imgs.detach())
            # netD loss
            d_loss = adversarial_loss(fake_pred, zeros) * 0.5 + \
                    adversarial_loss(real_pred, ones) * 0.5

            # information loss
            info_loss = lambda_cat * categorical_loss(fake_label, gt_labels) +\
                    lambda_con * continuous_loss(fake_code, code_input)

            # accumulate gradients
            info_loss.backward(retain_graph=True)
            d_loss.backward(inputs=netD_params, retain_graph=True)
            g_loss.backward(inputs=netG_params)

            optimizerG.step()
            optimizerD.step()
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch, epochs, i, len(loader),
                     d_loss.item(), g_loss.item(), real_pred.mean().item(), fake_pred.mean().item()))


                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
            # Check how the netG is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(loader)-1)):
                with torch.no_grad():
                    fake = netG(*fixed_noise).detach().cpu()
                imgs = vutils.make_grid(fake, padding=2, normalize=True)
                print('iter %i' % iters)
                cv2.imshow('vasya', (imgs.permute(2, 1, 0).numpy() * 255).astype(numpy.uint8))
                cv2.waitKey(2000)
                state_dict = dict(discriminator=netD.state_dict(),
                                  generator=netG.state_dict())
                torch.save(state_dict, snapshot_path)

            iters += 1
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
