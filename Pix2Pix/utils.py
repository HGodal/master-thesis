import torch
import numpy as np
import nibabel as nib
from config import config
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.ndimage.filters import uniform_filter1d


plt.style.use('ggplot')


def update_loss_plot(real_loss_list, fake_loss_list):
    iters = len(real_loss_list)
    iters_list = range(iters)

    batch_separations = np.arange(0, iters, config.NUM_BATCHES)[1:]
    real_loss_list_avg = uniform_filter1d(
        real_loss_list, size=100, mode='nearest')
    fake_loss_list_avg = uniform_filter1d(
        fake_loss_list, size=100, mode='nearest')

    plt.figure()
    plt.vlines(x=batch_separations, ymin=0, ymax=1, colors='black',
               ls='--', lw=1, alpha=0.5, label='Epoch')
    plt.plot(iters_list, real_loss_list_avg, label='Real Loss')
    plt.plot(iters_list, fake_loss_list_avg, label='Fake Loss')
    plt.axis([None, None, 0, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Critic')
    plt.title('Real and Fake Loss')
    plt.legend()
    plt.savefig(f'graphs/loss_latest_{config.NAME}.png')
    plt.close()


def save_all_images(gen, loader):
    gen.eval()

    for x, _, name in loader:
        x = x.to(config.DEVICE)

        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5

            for i in range(len(x)):
                y_fake_npy = np.squeeze(y_fake[i].cpu().detach().numpy())
                nib.save(nib.Nifti1Image(y_fake_npy, np.eye(4)), f'generated/{config.NAME}_{name[i]}.nii.gz')

                y_fake_slice = y_fake[i, :, :, :, y_fake.shape[4]//2]
                save_image(y_fake_slice, f'generated_images/{config.NAME}_{name[i]}.png')

    gen.train()


def save_some_examples(gen, val_loader, folder, epoch):
    x, y, _ = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        y_fake = y_fake * 0.5 + 0.5

        x_slice = x[:4, :, :, :, x.shape[4]//2]
        y_slice = y[:4, :, :, :, y.shape[4]//2]
        y_fake_slice = y_fake[:4, :, :, :, y_fake.shape[4]//2]

        save_image(x_slice, folder + f'/input_{epoch}_{config.NAME}.png')
        save_image(y_slice, folder + f'/target_{epoch}_{config.NAME}.png')
        save_image(y_fake_slice, folder + f'/gen_{epoch}_{config.NAME}.png')

    gen.train()


def save_checkpoint(model, optimizer, filename='my_checkpoint.pth.tar'):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
