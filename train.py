import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms 

from codes import dataset, motion_encoder, utils

import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import tqdm
import logging

#TODO: Delete this later
import warnings
warnings.filterwarnings('ignore')


def main():
    #TODO: Put training params and hparams in a yaml file
    parser = ArgumentParser(description='Head pose estimation using the ResNet-18 network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--config', dest='config', help='Path to config file.',
          default='configs/pose300wlp.yml', type=str)
    
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # Image Transformations
    transformations = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Normalize(mean=configs['images']['mean'],
                                 std=configs['images']['std'])])
    
    # Dataset
    dataset_train = dataset.Pose300WLP(**configs['data_params'], transform=transformations, test=False)
    dataset_test = dataset.Pose300WLP(**configs['data_params'], transform=transformations, test=True)

    # Dataloaders
    train_loader = DataLoader(dataset_train, batch_size=configs['train_params']['batch_size'], shuffle=True, num_workers=configs['train_params']['num_workers'])
    test_loader = DataLoader(dataset_test, batch_size=configs['train_params']['batch_size'], shuffle=False, num_workers=configs['train_params']['num_workers'])

    #Model
    model = motion_encoder.ResNet18(in_channels=configs['images']['in_channels'], pretrained=True)
    model.to(device)

#     # Test Model and DLoader
#     for some_batch in tqdm.tqdm(train_loader):
#         pass
#     print("TrainLoader OK")
#     for some_batch in tqdm.tqdm(test_loader):
#         pass
#     print("TestLoader OK")
#     imgs, _, labels, _ = next(iter(test_loader))

#     _ = model(imgs.to(device))
#     print(imgs.size(), labels.size())
#     print("Model Tested and ready")

    # Loss and Optimizer
    criterion = nn.MSELoss().to(device=device)
    optimizer = Adam(model.parameters(), lr=configs['train_params']['lr'])

    # Training Loop
    model.train()
    train_loss=[]
    for epoch in tqdm.trange(configs['train_params']['num_epochs']):
      for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            img, _, labels, _ = batch
            img = img.to(device)
            labels = labels.to(device)
            pose = model(img)

            loss = criterion(pose, labels)
            loss.backward()
            optimizer.step()
      
      train_loss.append(loss.detach().item())

    # Eval
    model.eval()
    eval_loss=[]
    with torch.no_grad():
      i=0
      for i, batch in enumerate(tqdm.tqdm(test_loader)):
            img, _, labels, _ = batch
            img = img.to(device)
            labels = labels.to(device)
            pose = model(img)
            eval_loss.append(criterion(pose, labels).item())




    return train_loss, eval_loss

if __name__ == '__main__':
    
    main()