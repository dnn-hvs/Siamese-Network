
#Alexnet================================================================================================================================================================

python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze

python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze


python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze

python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=alexnet --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze


#VGG16====================================================================================================================================================================
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py fmri --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze

python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py fmri --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze


python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py meg --region=early --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze

python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 #no foveation, no freeze
python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 #no foveation, freese
python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --foveate #foveation, no freeze
python3 main.py meg --region=late --num_epochs 300  --batch_size=32 --lr=0.005 --arch=vgg16 --optim=sgd --num_workers=30 --num_freeze_layers=4 --foveate, #no foveation, no freeze

#VGG19==========================================================================================================================================================



