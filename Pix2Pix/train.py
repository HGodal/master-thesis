import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
import localtransforms
from config import config
import torch.optim as optim
from dataset import MRIDataset
from generator_model import Generator
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from discriminator_model import Discriminator
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_all_images, update_loss_plot


def create_dataloaders():
    transform = transforms.Compose(
        [
            localtransforms.Normalize(),
            localtransforms.Reshape(config.IMAGE_SIZE),
            localtransforms.ToTensor(),
            localtransforms.AddGrayChannel(),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)]),
        ]
    )

    full_dataset = MRIDataset(config.ROOT_DIR, config.LOWRES, config.HIGHRES, transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    config.NUM_BATCHES = len(train_loader)
    
    return train_loader, val_loader


if __name__ == '__main__':
    wandb.init(project='Pix2Pix', entity='...', name=config.NAME)
    torch.backends.cudnn.benchmark = True

    print(config)

    transform = transforms.Compose(
        [
            localtransforms.Reshape(config.IMAGE_SIZE),
            localtransforms.ToTensor(),
            localtransforms.AddGrayChannel(),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)]),
        ]
    )

    disc = Discriminator(in_channels=1).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LR, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LR)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LR)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    real_loss_list = []
    fake_loss_list = []

    train_loader, val_loader = create_dataloaders()

    if config.TRAIN:
        for epoch in range(config.NUM_EPOCHS):
            loop = tqdm(train_loader, leave=True)

            for idx, (x, y, _) in enumerate(loop):
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                with torch.cuda.amp.autocast():
                    y_fake = gen(x)
                    D_real = disc(x, y)
                    D_real_loss = BCE(D_real, torch.ones_like(D_real))
                    D_fake = disc(x, y_fake.detach())
                    D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                    D_loss = (D_real_loss + D_fake_loss) / 2

                opt_disc.zero_grad()
                d_scaler.scale(D_loss).backward()
                d_scaler.step(opt_disc)
                d_scaler.update()

                with torch.cuda.amp.autocast():
                    D_fake = disc(x, y_fake)
                    G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                    L1 = L1_LOSS(y_fake, y) * config.L1_LAMBDA
                    G_loss = G_fake_loss + L1

                opt_gen.zero_grad()
                g_scaler.scale(G_loss).backward()
                g_scaler.step(opt_gen)
                g_scaler.update()

                if idx % 10 == 0:
                    loop.set_postfix(
                        D_real=torch.sigmoid(D_real).mean().item(),
                        D_fake=torch.sigmoid(D_fake).mean().item(),
                    )

                wandb.log({
                    "D Real": torch.sigmoid(D_real).mean().item(),
                    "D Fake": torch.sigmoid(D_fake).mean().item(),
                    "D Loss": D_loss.item(),

                    "G_fake_loss": G_fake_loss.item(),
                    "L1 Loss": L1.item(),
                    "G Loss": G_loss.item(),
                })

                real_loss_list.append(torch.sigmoid(D_real).mean().item())
                fake_loss_list.append(torch.sigmoid(D_fake).mean().item())

            update_loss_plot(real_loss_list, fake_loss_list)

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

            save_some_examples(gen, val_loader, 'evaluation', epoch)

    full_dataset = MRIDataset(config.ROOT_DIR, config.LOWRES, config.HIGHRES, transform, False)
    full_dataset_loader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    save_all_images(gen, full_dataset_loader)
