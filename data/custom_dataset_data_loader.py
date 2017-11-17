#import torch.utils.data
from data.base_data_loader import BaseDataLoader

## 3D Change add video Dataset
def CreateDataset(opt):
    dataset = None
    #print("================="+str(opt.dataset_mode))
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    ## add 3D videodataset loader
    elif opt.dataset_mode == 'v':
        from data.video_data import VideoDataset
        dataset = VideoDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batchSize,
        #     shuffle=not opt.serial_batches,
        #     num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataset

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
