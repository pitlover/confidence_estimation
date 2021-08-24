import os
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets
import excel
from sklearn.utils import gen_batches


def sort_get_loader(data, data_path, batch_size, idx, target, iscorrect, correctness, cur_confidence, cur_correctness, confidence, epoch, args):
    # dataset normalize values
    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
    elif data == 'svhn':
        mean = [0.5, 0.5, 0.5]
        stdv = [0.5, 0.5, 0.5]

    # augmentation
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # load datasets
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,
                                      transform=train_transforms,
                                      download=True)
    elif data == 'cifar10':  # cifar10_data /cifiar10_data
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
    elif data == 'svhn':
        train_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                  split='train',
                                  transform=train_transforms,
                                  download=True)

    # make Custom_Dataset
    if data == 'svhn':
        train_data = Sort_Custom_Dataset(train_set.data,
                                    train_set.targets, 'svhn',  idx, target, iscorrect, cur_confidence, cur_correctness, epoch, args, train_transforms)
    else:
        train_data = Sort_Custom_Dataset(train_set.data,
                                    train_set.targets, 'cifar',  idx, target, iscorrect, cur_confidence, cur_correctness, epoch, args, train_transforms)      ## x, y, idx

    # make DataLoader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4)


    print("-------------------Make loader-------------------")
    print("*** Sorting ***")
    print('Train Dataset :',len(train_loader.dataset))

    return train_loader

# Sort_Custom_Dataset class
class Sort_Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, idx, target, iscorrect, cur_confidence, cur_correctness, epoch, args, transform=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform
        self.iscorrect= iscorrect
        self.correctness = cur_correctness
        self.confidence = cur_confidence / epoch
        self.idx = idx

        if args.sort_mode == 0:
            acc_conf = self.correctness / self.confidence
            sort_index = acc_conf.argsort()
        elif args.sort_mode == 1:
            sort_index = self.confidence.argsort()
        elif args.sort_mode == 2:
            acc_conf = self.correctness / self.confidence
            sort_index = acc_conf.argsort()[::-1]
        elif args.sort_mode == 3:
            sort_index = self.confidence.argsort()[::-1]

        idx_list = []
        for a in sort_index:
            idx_list.append(idx[a])

        self.idx = idx_list

        self.x_data = np.array(self.x_data)[self.idx]
        self.y_data = np.array(self.y_data)[self.idx]

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'svhn':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))
        x = self.transform(img)
        return x, self.y_data[idx], idx

'''
# Sort_Custom_Dataset class
class Sort_Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, idx, target, iscorrect, confidence, epoch, transform=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform
        self.iscorrect= iscorrect
        self.confidence = confidence

        negative_index = (iscorrect == 0).nonzero()[0]
        positive_index = (iscorrect == 1).nonzero()[0] ## total 에서 correect 한 index
        print("negative : positive =", len(negative_index), len(positive_index))

        ### negative
        negative_idx = []  ## incorrect 한 paths
        for a in negative_index:
           negative_idx.append(idx[a])

        negative_target = target[negative_index]  ## incorrect 한 y_true
        negative_conf = confidence[negative_index]  ## incorrect 한 conf
        sort_negative_index = np.argsort(negative_conf)[::-1]
        # sort_negative_index = np.argsort(negative_conf)[::-1]
        sort_negative_conf = negative_conf[sort_negative_index]
        sort_negative_target = negative_target[sort_negative_index]
        sort_negative_idx = []
        for a in sort_negative_index:
            sort_negative_idx.append(negative_idx[a])
        print("**** Negative ***")
        print(sort_negative_conf[:5])
        print(sort_negative_conf[-5:-1])


        ### positive
        positive_idx = []  ## incorrect 한 paths
        for a in positive_index:
            positive_idx.append(idx[a])
        positive_target = target[positive_index]  ## incorrect 한 y_true
        positive_conf = confidence[positive_index]  ## incorrect 한 conf
        sort_positive_index = np.argsort(positive_conf)[::-1]
        sort_positive_conf = positive_conf[sort_positive_index]
        sort_positive_target = positive_target[sort_positive_index]
        sort_positive_idx = []
        for a in sort_positive_index:
            sort_positive_idx.append(positive_idx[a])
        print("**** Positive ***")
        print(sort_positive_conf[:5])
        print(sort_positive_conf[-5:-1])

        # # idx -> 배치 간격으로 sort
        # final = []
        # if len(sort_positive_idx) < len(sort_negative_idx):     # 학습 초반
        #     for a in range(len(sort_positive_idx)):
        #         final.append(sort_positive_idx[a])
        #         final.append(sort_negative_idx[a])
        #
        #
        #     final += sort_negative_idx[len(sort_positive_idx):]
        #     self.idx = final
        #     print("pos < neg")
        # else:
        #     final = sort_positive_idx[:len(sort_positive_idx)-len(sort_negative_idx)]# 학습 후반
        #     for a in range(len(sort_negative_idx)):
        #         final.append(sort_positive_idx[a + len(sort_positive_idx) - len(sort_negative_idx)])
        #         final.append(sort_negative_idx[a])
        #
        #     self.idx = final
        #     print("pos >= neg")


        self.idx = sort_positive_idx + sort_negative_idx
        print("pos -> neg")

        # self.idx = sort_negative_idx + sort_positive_idx
        # print("neg -> pos")

        self.x_data = np.array(self.x_data)[self.idx]
        self.y_data = np.array(self.y_data)[self.idx]

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'svhn':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))
        x = self.transform(img)
        return x, self.y_data[idx], idx
'''


def get_valid_loader(data, data_path, batch_size, validation_ratio = 0.1):
    # dataset normalize values
    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
    elif data == 'svhn':
        mean = [0.5, 0.5, 0.5]
        stdv = [0.5, 0.5, 0.5]

    # augmentation
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # load datasets
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,
                                      transform=train_transforms,
                                      download=True)
        valid_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=True,
                                     transform=test_transforms,
                                     download=True)

        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,
                                     transform=test_transforms,
                                     download=True)
    elif data == 'cifar10':  # cifar10_data /cifiar10_data
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=train_transforms,
                                     download=True)

        valid_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=test_transforms,
                                     download=False)

        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,
                                    transform=test_transforms,
                                    download=False)
    elif data == 'svhn':
        train_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                  split='train',
                                  transform=train_transforms,
                                  download=True)
        test_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                 split='test',
                                 transform=test_transforms,
                                 download=True)

    # make Custom_Dataset
    if data == 'svhn':
        train_data = Custom_Dataset(train_set.data,
                                    train_set.labels,
                                    'svhn', train_transforms)
        test_data = Custom_Dataset(test_set.data,
                                   test_set.labels,
                                   'svhn', test_transforms)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.labels)
        test_label = test_set.labels
    else:
        num_train = len(train_set)
        # Hard
        # indices = excel.extract_index()[:int(validation_ratio * num_train)]
        # Easy
        indices = excel.extract_index()[int(num_train - validation_ratio * num_train):]
        # Random
        # indices = list(range(num_train))
        split = int(np.floor(validation_ratio * len(indices)))

        random_seed= 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_data = Custom_Dataset(train_set.data,
                                    train_set.targets,
                                    'cifar', train_transforms)      ## x, y, idx
        valid_data = Custom_Dataset(valid_set.data,
                                    valid_set.targets,
                                    'cifar', test_transforms)  ## x, y, idx

        test_data = Custom_Dataset(test_set.data,
                                   test_set.targets,
                                   'cifar', test_transforms)


    # make DataLoader

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=4)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=valid_sampler,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)
    test_onehot = one_hot_encoding(test_set.targets)
    test_label = test_set.targets

    print("-------------------Make loader-------------------")
    # print('Train Dataset :',len(train_loader.dataset),
    #       '   Valid Dataset :', len(valid_loader.dataset),
    #       '   Test Dataset :',len(test_loader.dataset))
    print('Train Dataset :', len(train_sampler),
          '   Valid Dataset :', len(valid_sampler),
          '   Test Dataset :', len(test_loader.dataset))

    return train_loader, valid_loader, test_loader, test_onehot, test_label

def get_loader(data, data_path, batch_size):
    # dataset normalize values
    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
    elif data == 'svhn':
        mean = [0.5, 0.5, 0.5]
        stdv = [0.5, 0.5, 0.5]

    # augmentation
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # load datasets
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,
                                      transform=train_transforms,
                                      download=True)
        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,
                                     transform=test_transforms,
                                     download=False)
    elif data == 'cifar10':  # cifar10_data /cifiar10_data
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,
                                    transform=test_transforms,
                                    download=False)
    elif data == 'svhn':
        train_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                  split='train',
                                  transform=train_transforms,
                                  download=True)
        test_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                 split='test',
                                 transform=test_transforms,
                                 download=True)

    # make Custom_Dataset
    if data == 'svhn':
        train_data = Custom_Dataset(train_set.data,
                                    train_set.labels,
                                    'svhn', train_transforms)
        test_data = Custom_Dataset(test_set.data,
                                   test_set.labels,
                                   'svhn', test_transforms)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.labels)
        test_label = test_set.labels

        train_onehot = one_hot_encoding(train_set.labels)
        train_label = train_set.labels

    else:
        train_data = Custom_Dataset(train_set.data,
                                    train_set.targets,
                                    'cifar', train_transforms)      ## x, y, idx
        test_data = Custom_Dataset(test_set.data,
                                   test_set.targets,
                                   'cifar', test_transforms)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.targets)
        test_label = test_set.targets

        train_onehot = one_hot_encoding(train_set.targets)
        train_label = train_set.targets

    # make DataLoader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    print("-------------------Make loader-------------------")
    print('Train Dataset :',len(train_loader.dataset),
          '   Test Dataset :',len(test_loader.dataset))

    return train_loader, train_onehot, train_label, test_loader, test_onehot, test_label
# Custom_Dataset class
class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar':
            idx = int(idx)
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'svhn':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))

        x = self.transform(img)

        return x, self.y_data[idx], idx

def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot

