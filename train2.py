from dcgan import *
import os
import cv2
import numpy
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
from train1 import tiny_imagenet


def main():
    iters = 0
    epochs = 50
    snapshot_path = 'dcgan.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_list = []
    G_losses = []
    D_losses = []
    dataset = tiny_imagenet()
    batch_size = 20
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the generator
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    if os.path.exists(snapshot_path):
        state_dict = torch.load(snapshot_path)
        netG.load_state_dict(state_dict['generator'])
        netD.load_state_dict(state_dict['discriminator'])

    lr = 0.0001
    beta1 = 0.5
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

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

            output_data = netD(data.float().to(device)).view(-1)
            errD_real = criterion(output_data, ones)
            D_x = output_data.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            output_fake = netD(fake).view(-1)
            D_G_z1 = output_fake.mean().item()
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output_fake, zeros)
            errD = errD_real + errD_fake
            errD.backward(inputs=list(netD.parameters()), retain_graph=True)
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # for generator target is 1
            # Calculate G's loss based on this output
            errG = criterion(output_fake, ones)
            errG.backward(inputs=list(netG.parameters()))
            assert netG.main[0].weight.grad is not None
            optimizerG.step()
            optimizerD.step()
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch, epochs, i, len(loader),
                     errD.item(), errG.item(), D_x, D_G_z1))


                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
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
