import os
from copy import deepcopy

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import DataLoader as GMDataLoader

from utils.logging import QLogger
from utils.vis import visualize_log


class KFoldHelper:
    """Split data for (Stratified) K-Fold Cross-Validation."""

    def __init__(self,
                 n_splits=5,
                 stratify=False, graph=False):
        super().__init__()
        self.n_splits = n_splits
        self.stratify = stratify
        self.graph = graph

    def __call__(self, data):
        if self.stratify:
            labels = data.get_data_labels()
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        else:
            labels = None
            splitter = KFold(n_splits=self.n_splits, shuffle=True)
        #print(type(splitter), flush=True)
        #exit(0)
        n_samples = len(data)
        loader_class = DataLoader if not self.graph else GMDataLoader
        wrapper_class = _WrappedDataset if not self.graph else _WrappedGraphDataset
        for train_idx, val_idx in splitter.split(X=range(n_samples), y=labels):
            _train = Subset(data, train_idx)
            train_dataset = wrapper_class(_train, data.train_transform)
            train_loader = loader_class(dataset=train_dataset,
                                      batch_size=data.batch_size,
                                      shuffle=True,
                                      num_workers=data.num_workers, pin_memory=False)

            _val = Subset(data, val_idx)
            val_dataset = wrapper_class(_val, data.val_transform)
            val_loader = loader_class(dataset=val_dataset,
                                    batch_size=data.batch_size,
                                    shuffle=False,
                                    num_workers=data.num_workers, pin_memory=False)

            yield train_loader, val_loader


class _WrappedDataset:
    """Allows to add transforms to a given Dataset."""

    def __init__(self,
                 dataset,
                 transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

class _WrappedGraphDataset:
    """Allows to add transforms to a given Dataset."""

    def __init__(self,
                 dataset,
                 transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #sample, label = self.dataset[idx]
        #if self.transform is not None:
        #    sample = self.transform(sample)
        return self.dataset[idx]#sample, label

class CustomDataset:
    """Cats & dogs toy dataset."""

    def __init__(self, train_dataset, test_dataset, train_transform, test_transform,
                 num_workers: int = 16,
                 batch_size: int = 32):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__train_transform = train_transform
        self.__test_transform = test_transform
        #print(len(train_dataset), flush=True)

    def get_data_labels(self):
        dataset = self.train_dataset
        return [int(sample[1]) for sample in dataset]

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx: int):
        sample, label = self.train_dataset[idx]
        return sample, label

    @property
    def train_transform(self):
        return self.__train_transform

    @property
    def val_transform(self):
        return self.__test_transform

    @property
    def test_transform(self):
        return self.__test_transform


class CustomGraphDataset:
    """Cats & dogs toy dataset."""

    def __init__(self, train_dataset, test_dataset, train_transform, test_transform,
                 num_workers: int = 16,
                 batch_size: int = 32):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__train_transform = train_transform
        self.__test_transform = test_transform
        #print(len(train_dataset), flush=True)

    def get_data_labels(self):
        dataset = self.train_dataset
        print('len:', len(dataset[0].y[0]), flush=True)

        if len(dataset[0].y[0]) > 1:
            #print('x', flush=True)
            #print([sample.y[0].detach().numpy() for sample in dataset], flush=True)
            return [sample.y[0].detach().numpy() for sample in dataset] #multicalss
        else:
            #print('y', flush=True)
            #exit(0)
            return [sample.y.item() for sample in dataset] #binary

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx: int):
        return self.train_dataset[idx]

    @property
    def train_transform(self):
        return self.__train_transform

    @property
    def val_transform(self):
        return self.__test_transform

    @property
    def test_transform(self):
        return self.__test_transform

class CV:
    """(Stratified) K-Fold Cross-validation wrapper for a Trainer."""

    def __init__(self,
                 trainer,
                 config,
                 n_splits=5,
                 stratify=False):
        super().__init__()
        self.trainer = trainer
        self.n_splits = n_splits
        self.stratify = stratify
        self.config = config
        self.qlogger = QLogger()

    def fit(self, model, data):
        if os.path.exists("output/logs/" + self.qlogger.get_log_name()):
            os.remove("output/logs/" + self.qlogger.get_log_name())

        if self.n_splits <= 1:
            if self.config['MODEL']['DATASET'] == 'imagenet':
                train_loader = DataLoader(dataset=data[0],
                                          batch_size=self.config.getint('MODEL', 'BATCH_SIZE'),
                                          shuffle=True,
                                          num_workers=self.config.getint('SYSTEM', 'CPUS'), pin_memory=False)
                test_loader = DataLoader(dataset=data[1],
                                         batch_size=self.config.getint('MODEL', 'BATCH_SIZE'),
                                         shuffle=False,
                                         num_workers=self.config.getint('SYSTEM', 'CPUS'), pin_memory=False)
            elif self.config['MODEL']['DATASET'] in ['ogbg-molhiv', 'cora', 'ogbg-molpcba']:
                print('data', len(data.train_dataset), len(data.test_dataset), flush=True)
                train_loader = GMDataLoader(dataset=_WrappedGraphDataset(data.train_dataset, data.train_transform),
                                          batch_size=self.config.getint('MODEL', 'BATCH_SIZE'),
                                          shuffle=True,
                                          num_workers=self.config.getint('SYSTEM', 'CPUS'), pin_memory=False)
                test_loader = GMDataLoader(dataset=_WrappedGraphDataset(data.test_dataset, data.test_transform),
                                         batch_size=self.config.getint('MODEL', 'BATCH_SIZE'),
                                         shuffle=False,
                                         num_workers=self.config.getint('SYSTEM', 'CPUS'), pin_memory=False)
            else:
                train_loader = DataLoader(dataset=_WrappedDataset(data.train_dataset, data.train_transform),
                                          batch_size=data.batch_size,
                                          shuffle=True,
                                          num_workers=data.num_workers, pin_memory=False)
                test_loader = DataLoader(dataset=_WrappedDataset(data.test_dataset, data.test_transform),
                                         batch_size=data.batch_size,
                                         shuffle=False,
                                         num_workers=data.num_workers, pin_memory=False)

            self.trainer.fit(model, train_loader, test_loader)
            logs_per_epoch = len(model.qlogger.val_acc) // self.config.getint("MODEL", "MAX_EPOCHS")
            acc = sum(model.qlogger.val_acc[-logs_per_epoch:])/logs_per_epoch #model.qlogger.val_acc[-1]

            #print('ACC', sum(model.qlogger.val_acc[-logs_per_epoch:])/logs_per_epoch, flush=True)
            #print(self.config["LOGGING"]["VISUALIZATION"])
            if self.config.getboolean("LOGGING","VISUALIZATION"):
                model.qlogger.save()
                visualize_log(model.qlogger.get_log_name())
            return model, acc

        graph = "gcn" in self.qlogger.get_log_name().lower() or "gin" in self.qlogger.get_log_name().lower()
        #print(graph, flush=True)
        split_func = KFoldHelper(n_splits=self.n_splits, stratify=self.stratify, graph=graph)
        cv_data = split_func(data)
        loader_class = DataLoader if not graph else GMDataLoader
        wrapper_class = _WrappedDataset if not graph else _WrappedGraphDataset
        test_loader = loader_class(dataset=wrapper_class(data.test_dataset, data.test_transform),
                                 batch_size=data.batch_size,
                                 shuffle=False,
                                 num_workers=data.num_workers, pin_memory=False)
        best_model = None
        best_acc = 0
        best_fold_idx = -1
        for fold_idx, loaders in enumerate(cv_data):
            print(fold_idx, flush=True)
            print("[{}/{}] fold cross validation training".format(fold_idx + 1, self.n_splits))
            # Clone model & trainer:
            _model = deepcopy(model)
            _model.fold_idx = fold_idx
            _trainer = deepcopy(self.trainer)
            # Fit:
            _trainer.fit(_model, *loaders)
            pred = _trainer.test(_model, test_loader)
            print(pred[0], flush=True)
            if pred[0]['val_acc'] > best_acc:
                best_acc = pred[0]['val_acc']
                best_model = _model
                best_fold_idx = fold_idx

        if self.config.getboolean("LOGGING","VISUALIZATION"):
            best_model.qlogger.save()
            visualize_log(best_model.qlogger.get_log_name())
        return best_model, best_acc
