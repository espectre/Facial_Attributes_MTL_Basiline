import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset.CelebA import CelebA
from model.resnet import resnet50
import os
from torch.autograd import Variable
import argparse
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model', type=str,default='ckp/model_naive.pth')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
testset = CelebA('./data/list_eval_partition.txt', './data/list_attr_celeba.txt', '2',
                  './data/img_align_celeba/', transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

if not os.path.exists(opt.model):
    print('model doesnt exits')
    exit(1)

resnet=resnet50(pretrained=True)
resnet.fc=nn.Linear(2048,40)
resnet.load_state_dict(torch.load('ckp/model_naive.pth'))
resnet.cuda()

resnet.eval()
correct = torch.FloatTensor(40).fill_(0)
total = 0
with torch.no_grad():
    for batch_idx, (images, attrs) in enumerate(testloader):
        images = Variable(images.cuda())
        attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
        output = resnet(images)
        com1 = output > 0
        com2 = attrs > 0
        correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
        total += attrs.size(0)
print(correct / total)
print(torch.mean(correct / total))
