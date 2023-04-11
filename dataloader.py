from os.path import join
import os
import torch
from torch.utils.data import Dataset
from skimage import io as skimage


def matRead(data):
    data=data.transpose(2,0,1)/2047.
    data=torch.from_numpy(data)
    return data
''
class Dataset(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.panpath = path + 'SimulatedRawPAN/'
        self.mspath = path + 'SimulatedRawMS/'
        self.refpath = path + 'ReferenceRawMS/'

        files = os.listdir(self.panpath)
        img_list = []
        for index in files:
            num = index.split('p')[0]
            # num = re.split('[N|.]',index)[1]
            img_list.append(num)
        self.img_list = img_list

    def __getitem__(self, index):
        # print(index)
        fn = self.img_list[index]

        panBatch = skimage.imread(self.panpath + str(fn) + 'p.tif')
        # panBatch=sio.loadmat(self.panpath +'PAN'+ str(fn) + '.mat')['PAN']
        panBatch = panBatch[:, :, None]
        panBatch = matRead(panBatch)

        msBatch = skimage.imread(self.mspath + str(fn) + 'ms.tif')
        # msBatch = sio.loadmat(self.mspath + 'LRMS' + str(fn) + '.mat')['LRMS']
        msBatch = matRead(msBatch)
        # msBatch = torch.nn.functional.interpolate(msBatch, size=(256, 256), mode='bilinear')

        gtBatch = skimage.imread(self.refpath + str(fn) + 'MSref.tif')
        # gtBatch = sio.loadmat(self.refpath + 'HRMS' + str(fn) + '.mat')['HRMS']
        gtBatch = matRead(gtBatch)

        return gtBatch, msBatch, panBatch

    def __len__(self):
        return len(self.img_list)

def get_training_set(train_dir):
    # train_dir = join(root_dir, "train/")
    return Dataset(train_dir)

def get_val_set(val_dir):
    # val_dir = join(root_dir, "val/")
    return Dataset(val_dir)