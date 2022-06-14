import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import MRIDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import localtransforms as localtransforms
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix


plt.style.use('ggplot')
plot_extension = 'gen'
g = torch.Generator()
g.manual_seed(0)


def save_acc_loss(accuracy_stats, loss_stats):
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={'index':'epochs'})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={'index':'epochs'})
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
    sns.lineplot(data=train_val_acc_df, x = 'epochs', y='value', hue='variable',  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = 'epochs', y='value', hue='variable', ax=axes[1]).set_title('Train-Val Loss/Epoch')

    plt.savefig(f'plots/acc_loss_{plot_extension}.png', dpi=400)
    plt.close()


def save_report_confmat(y_pred_list, y_true_list):
    conf_mat_labels = {0: 'CN', 1: 'MCI', 2: 'AD'}

    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]

    conf_mat = confusion_matrix(y_true_list, y_pred_list)
    report = classification_report(y_true_list, y_pred_list)
    confusion_matrix_df = pd.DataFrame(conf_mat).rename(columns=conf_mat_labels, index=conf_mat_labels)

    _, ax = plt.subplots(figsize=(7,5))         
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax)

    plt.savefig(f'plots/confusion_{plot_extension}.png', dpi=400)
    plt.close()

    return report, conf_mat


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(img_folder, img_size, batch_size):
    transform = transforms.Compose(
        [
            localtransforms.Normalize(),
            localtransforms.Reshape(164),
            localtransforms.Crop(img_size),
            transforms.ToTensor(),
            localtransforms.AddGrayChannel(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    full_dataset = MRIDataset(img_folder, transform)
    train_size = int(0.6 * len(full_dataset))
    valid_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size

    split_size = [train_size, valid_size, test_size]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, split_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    return train_loader, val_loader, test_loader
