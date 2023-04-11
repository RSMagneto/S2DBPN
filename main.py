import argparse
import modelnew as model
import torch
import torch.nn as nn
import functions
import time
import os
import copy
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='')#, default='train/')
    parser.add_argument('--test_dir', help='testing_data',default='')#, default='test/')
    parser.add_argument('--outputs_dir',help='output model dir', default='')#, default='model/')
    parser.add_argument('--channels',help='numble of image channel', default=4)
    parser.add_argument('--batchSize', default=1)
    parser.add_argument('--testBatchSize', default=1)
    parser.add_argument('--epoch', default=1000)#1000
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device',default=torch.device('cuda:1'))
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr',type=float,default=0.00003,help='G‘s learning rate')
    parser.add_argument('--gamma',type=float,default=0.01,help='scheduler gamma')
    opt = parser.parse_args()

    seed = random.randint(1, 10000)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    train_set = dataloader.get_training_set(opt.input_dir)
    val_set = dataloader.get_val_set(opt.test_dir)

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    # 网络初始化：
    net = model.Net().to(opt.device)
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    # 建立优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[1600],gamma=opt.gamma)

    # loss=torch.nn.MSELoss()
    loss = torch.nn.L1Loss()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        net = net.to(opt.device)
        loss = loss.to(opt.device)
    best_weights = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_SAM=1.0
    for i in range(opt.epoch):
        # train
        net.train()
        epoch_losses = functions.AverageMeter()
        batch_time = functions.AverageMeter()
        end = time.time()
        for batch_idx, (gtBatch, msBatch, panBatch) in enumerate(train_loader):

            if torch.cuda.is_available():
                msBatch, panBatch, gtBatch = msBatch.to(opt.device), panBatch.to(opt.device), gtBatch.to(opt.device)
                msBatch = Variable(msBatch.to(torch.float32))
                panBatch = Variable(panBatch.to(torch.float32))
                gtBatch = Variable(gtBatch.to(torch.float32))
            N = len(train_loader)
            net.zero_grad()
            out = net(msBatch, panBatch)
            outLoss = loss(out, gtBatch)
            outLoss.backward(retain_graph=True)
            optimizer.step()
            epoch_losses.update(outLoss.item(), msBatch.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_idx + 1) % 100 == 0:
                training_state = '  '.join(
                    ['Epoch: {}', '[{} / {}]', 'mseLoss: {:.6f}']
                )
                training_state = training_state.format(
                    i, batch_idx, N, outLoss
                )
                print(training_state)

        print('%d epoch: loss is %.6f, epoch time is %.4f' % (i, epoch_losses.avg, batch_time.avg))
        if i % 25 == 0:
            torch.save(net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(i)))
        net.eval()
        epoch_SAM=functions.AverageMeter()
        with torch.no_grad():
            for j, (gtTest, msTest, panTest) in enumerate(val_loader):
                if torch.cuda.is_available():
                    msTest, panTest, gtTest = msTest.to(opt.device), panTest.to(opt.device), gtTest.to(opt.device)
                    msTest = Variable(msTest.to(torch.float32))
                    panTest = Variable(panTest.to(torch.float32))
                    gtTest = Variable(gtTest.to(torch.float32))
                    net = net.to(opt.device)
                mp = net(msTest, panTest)
                test_SAM=functions.SAM(mp, gtTest)
                if test_SAM==test_SAM:
                    epoch_SAM.update(test_SAM,msTest.shape[0])
            print('eval SAM: {:.6f}'.format(epoch_SAM.avg))

        if epoch_SAM.avg < best_SAM:
            best_epoch = i
            best_SAM = epoch_SAM.avg
            best_weights = copy.deepcopy(net.state_dict())
        print('best epoch:{:.0f}'.format(best_epoch))
        if i % 25 == 0:
            torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))
        scheduler.step()

    print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))