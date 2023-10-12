import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import einops
from .uci_data import get_regression_data, get_classification_data
from sklearn import preprocessing
import sklearn.datasets


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def extract_classes(dataset, classes):
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for target in classes:
        idx = idx | (dataset.targets == target)

    data, targets = dataset.data[idx], dataset.targets[idx]
    return data, targets


def getDataset(dataset, transform_train, transform_test=None, uci_regression=True, normalize=False):
    transform_split_mnist = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    if transform_test is None:
        transform_test = transform_train

    if (dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        # trainset = transform_dataset(trainset)
        # testset = transform_dataset(testset)
        num_classes = 10
        img, _ = trainset[0]
        inputs = img.size(0)

    elif (dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        # trainset = transform_dataset(trainset)
        # testset = transform_dataset(testset)
        num_classes = 100
        img, _ = trainset[0]
        inputs = img.size(0)

    elif (dataset == 'SVHN'):
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        # trainset = transform_dataset(trainset)
        # testset = transform_dataset(testset)
        num_classes = 10
        img, _ = trainset[0]
        inputs = img.size(0)

    elif (dataset == 'STL10'):
        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
        # trainset = transform_dataset(trainset)
        # testset = transform_dataset(testset)
        num_classes = 10
        img, _ = trainset[0]
        inputs = img.size(0)

    elif (dataset == 'MNIST'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        inputs = 1

    elif 'mix' in dataset:
        label_files = np.load(f'./data/mix_labels/{dataset}.npz')
        labels_train = label_files['labels_train']
        labels_test = label_files['labels_test']
        dataset_name = dataset.split('-')[0]
        if dataset_name == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            trainset.targets = list(labels_train)
            testset.targets = list(labels_test)
        elif dataset_name == 'STL10':
            trainset = torchvision.datasets.STL10(root='./data', split='train', download=True,
                                                  transform=transform_train)
            testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
            trainset.labels = labels_train
            testset.labels = labels_test
        elif dataset_name == 'SVHN':
            trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
            testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
            trainset.labels = labels_train
            testset.labels = labels_test
        else:
            raise NotImplementedError

        num_classes = 10
        inputs = 3

    elif (dataset == 'SplitMNIST-2.1'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [0, 1, 2, 3, 4])
        test_data, test_targets = extract_classes(testset, [0, 1, 2, 3, 4])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        inputs = 1

    elif (dataset == 'SplitMNIST-2.2'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [5, 6, 7, 8, 9])
        test_data, test_targets = extract_classes(testset, [5, 6, 7, 8, 9])
        train_targets -= 5  # Mapping target 5-9 to 0-4
        test_targets -= 5  # Hence, add 5 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        inputs = 1

    elif (dataset == 'SplitMNIST-5.1'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [0, 1])
        test_data, test_targets = extract_classes(testset, [0, 1])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif (dataset == 'SplitMNIST-5.2'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [2, 3])
        test_data, test_targets = extract_classes(testset, [2, 3])
        train_targets -= 2  # Mapping target 2-3 to 0-1
        test_targets -= 2  # Hence, add 2 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif (dataset == 'SplitMNIST-5.3'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [4, 5])
        test_data, test_targets = extract_classes(testset, [4, 5])
        train_targets -= 4  # Mapping target 4-5 to 0-1
        test_targets -= 4  # Hence, add 4 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif (dataset == 'SplitMNIST-5.4'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [6, 7])
        test_data, test_targets = extract_classes(testset, [6, 7])
        train_targets -= 6  # Mapping target 6-7 to 0-1
        test_targets -= 6  # Hence, add 6 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif (dataset == 'SplitMNIST-5.5'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        train_data, train_targets = extract_classes(trainset, [8, 9])
        test_data, test_targets = extract_classes(testset, [8, 9])
        train_targets -= 8  # Mapping target 8-9 to 0-1
        test_targets -= 8  # Hence, add 8 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif (dataset == 'cifar_c'):
        dataset = np.load('./data/cifar_val.npz')
        x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']
        x_train = einops.rearrange(x_train, "n h w c -> n c h w")
        x_test = einops.rearrange(x_test, "n h w c -> n c h w")
        x_train, x_test = torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float()
        y_train, y_test = torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()
        trainset = CustomDataset(x_train, y_train)
        testset = CustomDataset(x_test, y_test)
        num_classes = 10
        inputs = 3
    elif dataset == 'twomoon':
        ntrain = 500
        ntest = 200
        noise = 0.1
        x_train, y_train = sklearn.datasets.make_moons(ntrain, random_state=1, noise=noise)
        x_test, y_test = sklearn.datasets.make_moons(ntest, random_state=2, noise=noise)
        scaler_data = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler_data.transform(x_train)
        x_test = scaler_data.transform(x_test)
        x_train, x_test = torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float()
        y_train, y_test = torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()
        trainset = CustomDataset(x_train, y_train)
        testset = CustomDataset(x_test, y_test)
        num_classes = 2
        inputs = 2
    elif dataset == 'toy_regression':
        x_train = np.array(np.arange(-1, 1, 0.02))
        y_train = np.sin(x_train * np.pi) + 0.1 * np.random.randn(100)
        x_train = x_train.reshape(-1, 1)

        x_test = np.array(np.concatenate((np.arange(-2, -1, 0.02), np.arange(1, 2, 0.02))))
        y_test = np.sin(x_test) + 0.1 * np.random.randn(100)
        x_test = x_test.reshape(-1, 1)

        x_train, x_test = torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float()
        y_train, y_test = torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()
        trainset = CustomDataset(x_train, y_train)
        testset = CustomDataset(x_test, y_test)

        inputs = 1
        num_classes = 0
    else:
        if uci_regression:
            uci_dataset = get_regression_data(dataset)
            num_classes = 0
            inputs = uci_dataset.D - 1
        else:
            uci_dataset = get_classification_data(dataset)
            num_classes = uci_dataset.K
            inputs = uci_dataset.D - 1
        data, target = uci_dataset.read_data()
        ratio = 0.8
        indices = np.random.choice(data.shape[0], data.shape[0], replace=False)
        data, target = data[indices], target[indices]
        train_size = int(ratio * data.shape[0])
        data = data[:, 1:]  # added this to drop the enumeration from being used as a feature

        if normalize:
            scaler_data = preprocessing.StandardScaler().fit(data[:train_size])
            data = scaler_data.transform(data)
        if uci_regression:
            if normalize:
                scaler_target = preprocessing.StandardScaler().fit(target[:train_size].reshape(-1, 1))
                target = scaler_target.transform(target.reshape(-1, 1)).squeeze()
            train_data, train_targets = torch.from_numpy(data[:train_size]).float(), torch.from_numpy(
                target[:train_size]).float()
            test_data, test_targets = torch.from_numpy(data[train_size:]).float(), torch.from_numpy(
                target[train_size:]).float()
        else:
            target = target.squeeze()
            train_data, train_targets = torch.from_numpy(data[:train_size]).float(), torch.from_numpy(
                target[:train_size]).long()
            test_data, test_targets = torch.from_numpy(data[train_size:]).float(), torch.from_numpy(
                target[train_size:]).long()

        trainset = CustomDataset(train_data, train_targets)
        testset = CustomDataset(test_data, test_targets)

    return trainset, testset, inputs, num_classes


# in dlm-bnn and bnn-hydra
def getDataloader(trainset, testset, valid_size, batch_size, num_workers, split_train=True):

    if split_train:
        num_train = len(trainset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  num_workers=num_workers)

        return train_loader, valid_loader, test_loader
    else:
        num_test = len(testset)
        indices = list(range(num_test))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_test))
        test_idx, valid_idx = indices[split:], indices[:split]

        test_sampler = SubsetRandomSampler(test_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler,
                                                  num_workers=num_workers)

        return train_loader, valid_loader, test_loader


def getTransformedDataset(dataset, model_cfg, uci_regression=True, normalize=False):
    transform_train, transform_test = model_cfg.transform_train, model_cfg.transform_test
    if dataset == 'MNIST':
        transform_train = transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    return getDataset(dataset, transform_train, transform_test, uci_regression=uci_regression, normalize=normalize)