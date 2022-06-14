import torch
import random
import numpy as np
from glob import glob
import nibabel as nib
import localtransforms
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.list_files = sorted(glob(f'{root_dir}/*.nii.gz'))
        self.transform = transform

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_path = self.list_files[index]
        image = nib.load(img_path).get_fdata()

        if random() > 0.80:
            image = np.flip(image, 0)

        if random() > 0.80:
            image = np.flip(image, 1)

        if random() > 0.80:
            image = np.flip(image, 2)

        image = image.astype(np.float32)

        target = 0
        if 'MCI' in img_path:
            target = 1
        elif 'Dementia' in img_path:
            target = 2

        if self.transform:
            image = self.transform(image)

        return image, target


if __name__ == '__main__':
    transforms = transforms.Compose(
        [
            localtransforms.Normalize(),
            localtransforms.Reshape(164),
            localtransforms.Crop(128),
            transforms.ToTensor(),
            localtransforms.AddGrayChannel(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = MRIDataset('../data/nifti/dataset/highres', transform=transforms)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    all_targets = torch.tensor([])
    for x, target in loader:
        print(x.shape)
        print(target)
