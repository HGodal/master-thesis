import wandb
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from networks import Classifier
from utils import multi_acc, create_dataloaders, save_acc_loss, save_report_confmat


wandb.init(project='Classifier', entity='...', name='CNN')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

num_epochs = 31
batch_size = 2
learning_rate = 0.002
img_size = 128
in_channels = 1
num_classes = 3
img_folder = '../data/nifti/dataset'

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_loader, val_loader, test_loader = create_dataloaders(img_folder, img_size, batch_size)

model = Classifier(in_channels=in_channels, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

wandb.watch(model, log_freq=1)

for epoch in range(num_epochs):
    train_epoch_loss = 0
    train_epoch_acc = 0

    model.train()
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        y_train_pred = model(data)
        train_loss = criterion(y_train_pred, targets)
        train_acc = multi_acc(y_train_pred, targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        for batch_idx, (data, targets) in enumerate(tqdm(val_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)

            y_val_pred = model(data).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
            val_loss = criterion(y_val_pred, targets)
            val_acc = multi_acc(y_val_pred, targets)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

    print(
        f'Epoch {epoch+0:02}: | ' +
        f'Train Loss: {train_epoch_loss/len(train_loader):.5f} | ' +
        f'Val Loss: {val_epoch_loss/len(val_loader):.5f} | ' +
        f'Train Acc: {train_epoch_acc/len(train_loader):.3f}| ' +
        f'Val Acc: {val_epoch_acc/len(val_loader):.3f}'
    )

    wandb.log({
        'Train Loss': train_epoch_loss/len(train_loader),
        'Val Loss': val_epoch_loss/len(val_loader),
        'Train Acc': train_epoch_acc/len(train_loader),
        'Val Acc': val_epoch_acc/len(val_loader),
    })

    save_acc_loss(accuracy_stats, loss_stats)


y_pred_list = []
y_true_list = []

model.eval()
with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        y_test_pred = model(data)
        _, y_pred_tag = torch.max(y_test_pred, dim=1)

        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(targets.cpu().numpy())

report, conf_mat = save_report_confmat(y_pred_list, y_true_list)

print(report)
print(conf_mat)
