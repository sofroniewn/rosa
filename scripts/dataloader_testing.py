from typing import Optional

import torch
import zarr
from time import time, sleep


class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        super(ZarrDataset, self).__init__()

        self.path = path

        self.array = zarr.open(path, mode='r')

    def __len__(self) -> int:
        return self.array.shape[0]

    def __getitem__(self, idx):
        return self.array[idx]


class PtDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        super(PtDataset, self).__init__()

        self.path = path

    def __len__(self) -> int:
        return 1000

    def __getitem__(self, idx):
        return torch.load(self.path)['results']


class ZarrDataset2(torch.utils.data.Dataset):
    def __init__(self, path: str):
        super(ZarrDataset2, self).__init__()

        # self.path = path
        # self.array = None # type: Optional[zarr.Array]

    def _worker_init(self):
        self.array = zarr.open('/Users/nsofroniew/Documents/data/multiomics/enformer/scratch/example.zarr', mode='r')
        
    def __len__(self) -> int:
        return 1000 #self.array.shape[0]

    def __getitem__(self, idx):
        return self.array[idx]

def get_data(path: str) -> zarr.Array:
    return zarr.open(path, mode='r')


def worker_init_fn(worker_id):
    print('did this get called?')
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._worker_init()


if __name__ == "__main__":
    PT = '/Users/nsofroniew/Documents/data/multiomics/enformer/scratch'
    ds = ZarrDataset(PT + '/example.zarr')
    # dl = torch.utils.data.DataLoader(ds, shuffle=False, worker_init_fn=worker_init_fn, num_workers=4)
    # ds = PtDataset(PT + '/example_0.pt')
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    print('Starting')    
    start = time()
    for batch in iter(dl):
        # sleep(.005)
        pass
    stop = time()
    print(stop - start)    


