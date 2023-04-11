import skimage.io as skimage
import torch
import numpy as np
from numpy import *
import pdb
from sewar.full_ref import sam

def convert_image_np(inp,opt):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    inp=inp*2047.
    return inp
def convert_image_np_center(inp):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    inp=inp*2047.
    return inp
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img,hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.numpy()
    sr_img=sr_img[-1,:,:,:]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value
