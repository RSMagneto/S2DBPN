import modelnew as model
import torch
import functions
import numpy
import os
import argparse
import scipy.io as sio
import re

def test_matRead(data):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(torch.device('cuda:0')).type(torch.cuda.FloatTensor)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mspath', help='test lrms image name', default='')# default='mspath')
    parser.add_argument('--panpath', help='test hrpan image name', default='')# default='panpath')
    parser.add_argument('--modelpath', help='output model dir', default='')# default='model/best.pth')
    parser.add_argument('--saveimgpath', help='output model dir', default='')# default='result/')
    parser.add_argument('--device', default=torch.device('cuda:0'))
    opt = parser.parse_args()

    net = model.Net().to(opt.device)

    modelname = opt.modelpath
    net.load_state_dict(torch.load(modelname))
    num_params = sum(param.numel() for param in net.parameters())
    print("Number of parameter: %.2fM" %(num_params/1e6))
    with torch.no_grad():
        for msfilename in os.listdir(opt.mspath):
            # num = msfilename.split('m')[0]
            num=re.split('[S|.]',msfilename)[1]
            print(opt.mspath + msfilename)
            # ms_val = io.imread(opt.mspath + msfilename)#'lrms.tif'
            ms_val=sio.loadmat(opt.mspath + msfilename)['LRMS']#'lrms.mat'
            ms_val = test_matRead(ms_val)
            # panname = msfilename.split('m')[0]+'p.tif' #'pan.tif'
            panname='PAN'+str(num)+'.mat'
            # pval = io.imread(opt.panpath + panname)
            pan_val=sio.loadmat(opt.panpath + panname)['PAN']#'pan.mat'
            pan_val = pan_val[:, :, None]
            pan_val = test_matRead(pan_val)
            in_s = net(ms_val, pan_val)
            outname = opt.saveimgpath + num +'.mat'
            output=functions.convert_image_np(in_s.detach(), opt).astype(numpy.uint16)
            # io.imsave(outname, convert)
            sio.savemat(outname, {'result': output})

if __name__ == '__main__':
    main()
