from __future__ import print_function, division
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from skimage import transform, io
import os
import pandas as pd

from torch.utils.data import Dataset
import torch


class CambridgeDataset(Dataset):
    def __init__(self,
                 txt_name="dataset_train.txt",
                 contain_dir="G:\\Thesis\\Dataset\\Cambridge",
                 dataset='shopfacade',
                 transform=None,
                 test=False):
        if dataset == 'shopfacade':
            self.dir_path = os.path.join(contain_dir, "\\ShopFacade\\")
        elif dataset == 'kingscollege':
            self.dir_path = os.path.join(contain_dir, "\\kingscollege\\")
        elif dataset == 'street':
            self.dir_path = os.path.join(contain_dir, "\\street\\")
        elif dataset == 'oldhospital':
            self.dir_path = os.path.join(contain_dir, "\\oldhospital\\")
        elif dataset == 'greatcourt':
            self.dir_path = os.path.join(contain_dir, "\\greatcourt\\")
        elif dataset == 'stmaryschurch':
            self.dir_path = os.path.join(contain_dir, "\\stmaryschurch\\")
        else:
            raise ValueError('Check the name of the dataset!')
        self.text_path = self.dir_path + txt_name

        self.df = pd.read_csv(self.text_path, sep=' ', header=None, skiprows=3)
        self.img_names = self.df.iloc[:, 0]
        self.transform = transform
        # self.to_tensor = ToTensor()
        # self.to_pil = ToPILImage()

    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file
        :param name: name of targeted image
        :return: a PIL image
        """

        image = io.imread(os.path.join(self.dir_path, name))
        return image

    def get_poses_from_folder(self, index):
        return np.array(self.df.iloc[index, 1]).astype('float'), np.array(self.df.iloc[index, 2]).astype('float'), \
               np.array(self.df.iloc[index, 3]).astype('float'), \
               np.array(self.df.iloc[index, 4]).astype('float'), np.array(self.df.iloc[index, 5]).astype('float'), \
               np.array(self.df.iloc[index, 6]).astype('float'), np.array(self.df.iloc[index, 7]).astype('float')

    def __len__(self):
        """
        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set. Here we apply our pre-processing things like crop styles and
        subtractive color process using CMYK color model, generating edge-maps, etc.
        :param index: index of item in IDs list
        :return: a sample of data as a dict
        """
        # y_descreen = self.get_image_from_folder(self.img_names[index])
        # if torch.is_tensor(index):
        #     index = index.tolist()

        image = self.get_image_from_folder(self.img_names[index])
        x, y, z, w, p, q, r = self.get_poses_from_folder(index)
        sample = {'image': image, 'x': x, 'y': y, 'z': z, 'w': w, 'p': p, 'q': q, 'r': r}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, x, y, z, w, p, q, r = \
            sample['image'], sample['x'], sample['y'], sample['z'], sample['w'], sample['p'], sample['q'], sample['r']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'x': x, 'y': y, 'z': z, 'w': w, 'p': p, 'q': q, 'r': r}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, x, y, z, w, p, q, r = \
            sample['image'], sample['x'], sample['y'], sample['z'], sample['w'], sample['p'], sample['q'], sample['r']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'x': x, 'y': y, 'z': z, 'w': w, 'p': p, 'q': q, 'r': r}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, x, y, z, w, p, q, r = \
            sample['image'], sample['x'], sample['y'], sample['z'], sample['w'], sample['p'], sample['q'], sample['r']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'x': torch.from_numpy(np.asarray(x)), 'y': torch.from_numpy(np.asarray(y)),
                'z': torch.from_numpy(np.asarray(z)),
                'w': torch.from_numpy(np.asarray(w)), 'p': torch.from_numpy(np.asarray(p)),
                'q': torch.from_numpy(np.asarray(q)), 'r': torch.from_numpy(np.asarray(r))}


class CambridegDatasetLoader(Dataset):
    """
    The dataset class suitable for Solver()
    """
    def __init__(self,
                 image_path,
                 metadata_path,
                 transform,
                 num_val=100,
                 mode='train'):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[3:]

        print(self.lines.__len__())
        print(self.lines[0])

        self.test_filenames = []
        self.test_poses = []
        self.train_filenames = []
        self.train_poses = []

        for i, line in enumerate(self.lines):
            splits = line.split()
            filename = splits[0]
            values = splits[1:]
            values = list(map(lambda x: float(x.replace(",", "")), values))

            filename = os.path.join(self.image_path, filename)

            if self.mode == 'train':
                self.train_filenames.append(filename)
                self.train_poses.append(values)
            elif self.mode == 'test':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
            elif self.mode == 'val':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
                if i > num_val:
                    break
            else:
                assert 'Unavailable mode'

        self.num_train = self.train_filenames.__len__()
        self.num_test = self.test_filenames.__len__()
        print("Number of Train", self.num_train)
        print("Number of Train", self.num_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(self.train_filenames[index])
            pose = self.train_poses[index]
        elif self.mode in ['val', 'test']:
            image = Image.open(self.test_filenames[index])
            pose = self.test_poses[index]
        else:
            image = None
            # pose = 0
        return self.transform(image), torch.Tensor(pose)


def get_loader(model, image_path, metadata_path, mode, batch_size, is_shuffle=False, num_val=100):

    # Predefine image size
    if model == 'Googlenet':
        img_size = 300
        img_crop = 299
    elif model == 'Resnet':
        img_size = 256
        img_crop = 224

    if mode == 'train':
        cust_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_crop),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # metadata_path_val =  txt_name="G:\\Thesis\\Dataset\\Cambridge\\ShopFacade\\dataset_test.txt",
        datasets = {'train': CambridegDatasetLoader(image_path=image_path,
                                                    metadata_path=metadata_path,
                                                    mode='train',
                                                    transform=cust_transform,
                                                    num_val=num_val),
                    'val':  CambridegDatasetLoader(image_path=image_path,
                                                   metadata_path=metadata_path,
                                                   mode='val',
                                                   transform=cust_transform,
                                                   num_val=num_val)}
        data_loaders = {'train': DataLoader(datasets['train'], batch_size, is_shuffle, num_workers=4),
                        'val': DataLoader(datasets['val'], batch_size, is_shuffle, num_workers=4)}
    elif mode == 'test':
        cust_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        batch_size = 1
        is_shuffle = False
        dataset = CambridegDatasetLoader(image_path=image_path,
                                         metadata_path=metadata_path,
                                         mode='test',
                                         transform=cust_transform)
        data_loaders = DataLoader(dataset, batch_size, is_shuffle, num_workers=4)

    else:
        assert 'Unavailable Mode'
        data_loaders = None

    return data_loaders


if __name__ == "__main__":
    dataset = CambridgeDataset(transform=transforms.Compose([Rescale(256),
                                                             RandomCrop(224),
                                                             ToTensor()]))
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].size(),
              sample['x'], sample['y'], sample['z'],
              sample['w'], sample['p'], sample['q'], sample['r'])

        if i == 4:
            break
