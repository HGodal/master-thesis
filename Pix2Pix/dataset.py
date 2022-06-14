import os
import numpy as np
import nibabel as nib
import localtransforms
from random import random, choice
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MRIDataset(Dataset):
    def __init__(self, root_dir, lowres_dir, highres_dir, transform=None, augmentation=True):
        self.root_dir = root_dir
        self.lowres_dir = os.path.join(root_dir, lowres_dir)
        self.highres_dir = os.path.join(root_dir, highres_dir)

        self.list_files_lowres = sorted(os.listdir(self.lowres_dir))
        self.list_files_highres = sorted(os.listdir(self.highres_dir))
        self.augmentation = augmentation
        self.transform = transform

    def __len__(self):
        return len(self.list_files_lowres)

    def __getitem__(self, index):
        img_file_lowres = self.list_files_lowres[index]
        img_file_highres = self.list_files_highres[index]

        img_path_lowres = os.path.join(self.lowres_dir, img_file_lowres)
        img_path_highres = os.path.join(self.highres_dir, img_file_highres)

        input_image = nib.load(img_path_lowres).get_fdata().astype(np.float32)
        target_image = nib.load(img_path_highres).get_fdata().astype(np.float32)

        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min())


        if self.augmentation: 
            if random() > 0.80:
                input_image = np.flip(input_image, 0).copy()
                target_image = np.flip(target_image, 0).copy()

            if random() > 0.80:
                input_image = np.flip(input_image, 1).copy()
                target_image = np.flip(target_image, 1).copy()

            if random() > 0.80:
                input_image = np.flip(input_image, 2).copy()
                target_image = np.flip(target_image, 2).copy()

            if random() > 0.80:
                size = 10
                shiftlist = [0, size, 2*size]
                x_diff, y_diff, z_diff = choice(shiftlist), choice(shiftlist), choice(shiftlist)
                padding = ((size, size), (size, size), (size, size))

                x, y, z = input_image.shape
                input_image = np.pad(input_image, padding, mode='edge')
                input_image = input_image[x_diff:x+x_diff, y_diff:y+y_diff, z_diff:z+z_diff]

                x, y, z = target_image.shape
                target_image = np.pad(target_image, padding, mode='edge')
                target_image = target_image[x_diff:x+x_diff, y_diff:y+y_diff, z_diff:z+z_diff]
        
        input_image = input_image.astype(np.float32)
        target_image = target_image.astype(np.float32)
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        

        return input_image, target_image, img_file_lowres[:-11]


if __name__ == '__main__':
    transforms = transforms.Compose(
        [
            localtransforms.Reshape(256),
            localtransforms.ToTensor(),
            localtransforms.AddGrayChannel(),
            transforms.Normalize(
                [0.5 for _ in range(1)],
                [0.5 for _ in range(1)]),
        ]
    )

    dataset = MRIDataset('../data/nifti/dataset', 'lowres', 'highres', transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    for x, y, name in loader:
        print(x.shape, y.shape)
        exit()
