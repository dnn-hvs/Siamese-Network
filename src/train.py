import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim

import _init_paths

from dataset.siamese-network-dataset import SiameseNetworkDataset 
from network.siamese-network import SiameseNetwork
from loss.contrastive-loss import ContrastiveLoss
from utils import config, plot_images

# import SiameseNetworkDataset from siamese-network-dataset
# import SiameseNetwork from siamese-network
# import ContrastiveLoss from contrastive-loss





folder_dataset = dset.ImageFolder(root=config.Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
vis_dataloader = DataLoader(siamese_dataset,
shuffle=True,
num_workers=8,
batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
plot_images.imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=config.Config.train_batch_size)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,config.Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
plot_images.show_plot(counter,loss_history)