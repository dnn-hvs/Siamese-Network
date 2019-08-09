from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torchvision.datasets as dset


# from dataset.siames-network-dataset import SiameseNetworkDataset
from utils.config import Config
from utils.plot_images import imshow,show_plot
from network.siamese_network import SiameseNetwork
from dataset.siamese_network_dataset import SiameseNetworkDataset


folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)
net = SiameseNetwork().cuda()

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

