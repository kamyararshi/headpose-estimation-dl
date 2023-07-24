import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from codes import dataset, motion_encoder, utils

import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import tqdm
import logging
from time import gmtime, strftime
from shutil import copy
import os


#TODO: Delete this later
import warnings
warnings.filterwarnings('ignore')


def main():
    #TODO: Put training params and hparams in a yaml file
    parser = ArgumentParser(description='Head pose estimation using the ResNet-18 network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--config', dest='config', help='Path to config file.',
          default='./configs/pose300wlp.yml', type=str)
    parser.add_argument('--log_dir', dest='log_dir', help='Path to logs and saved checkpoints.',
          default='./logs/', type=str)
    
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # Log Dir
    log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
    log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

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

    #TODO: Add MultiStepLR later

    # Training Loop
    model.train()
    writer = SummaryWriter(log_dir=log_dir)
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
        writer.add_scalar("Train Loss", loss.detach().item(), epoch)


        # Eval
        model.eval()
        eval_loss=[]
        with torch.no_grad():
            i=0
            for i, batch in enumerate(test_loader):
                img, _, labels, _ = batch
                img = img.to(device)
                labels = labels.to(device)
                pose = model(img)
                eval_loss.append(criterion(pose, labels).item())
            
            # Plot the image and pose using the 'draw_axis' function
            pitch, yaw, roll, tdx, tdy, tdz, _ = pose[0].cpu()
            image_to_plot = utils.draw_axis(img[:1,...], yaw, pitch, roll, tdx, tdy)
            image_to_plot_tensor = torch.from_numpy(image_to_plot).permute(2,0,1)

            # Add the image to TensorBoard
            writer.add_image("Example Image", image_to_plot_tensor, global_step=epoch)
        
        # Logging the evaluation loss to TensorBoard
        avg_eval_loss = sum(eval_loss) / len(eval_loss)
        writer.add_scalar("Eval Loss", avg_eval_loss, epoch)

    writer.close()

    # Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss[-1],
            }, os.path.join(log_dir, f'{epoch+1}.pth.tar'))


if __name__ == '__main__':
    
    main()