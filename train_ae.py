import torch
import torch.nn as nn
import os 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import argparse
import wandb

from sketch_ae.datasets.celeba_sketch import CelebaSketch
from sketch_ae.models.Combine_AE import Combine_AE

def get_args_parser():

    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage')
    parser.add_argument('--dataset', type=str, default='./datasets/celeba_sketch/')
    parser.add_argument('--log_dir', type=str, default='./sketch_ae/outputs/')
    parser.add_argument('--log_rate', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=int, default=3e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=1)

    parser.add_argument('--checkpoint', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--date', type=str, default='1028')
    args = parser.parse_args()
    return args

def main():
    model = Combine_AE(img_size=args.img_size, img_channels=args.img_channels).to(args.device)
    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Trainable Parameters = %d" % (Num_Param))

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f'model loaded from {args.checkpoint}')

    run = wandb.init(
        project='deepfaceketch_ae',
        entity='ohicarip',
        # config=vars(args),
        name='ae_1028',
    )
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.MSELoss()

    transform = transforms.Compose([
                # transforms.RandomSizedCrop(64),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5,
                                    std=1)
                ])

    dataset = CelebaSketch(data_dir=args.dataset, transform=transform)
    img_inds = np.arange(len(dataset))

    train_inds = img_inds[:int(0.9 * len(img_inds))]
    test_inds = img_inds[int(0.9 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(train_inds),
        num_workers=0,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(test_inds),
        num_workers=0,
    )

    encoder_key = ['eye1', 'eye2', 'nose', 'mouth', 'face']

    for epoch in range(args.epochs):
        running_loss = {
            'eye1' : 0,
            'eye2' : 0,
            'nose' : 0,
            'mouth' : 0,
            'face' : 0
        }

        running_log = {
            'eye1' : [],
            'eye2' : [],
            'nose' : [],
            'mouth' : [],
            'face' : []    
        }

        model.train()
        for i, sketches in enumerate(train_loader):
            sketches = sketches.to(args.device)
            for k, key in enumerate(encoder_key):
                if k == 0:
                    target = sketches[:,:, 94:94+64, 54:54+64]
                elif k == 1:
                    target = sketches[:,:, 94:94+64, 128:128+64]
                elif k == 2:
                    target = sketches[:,:, 116:116+96, 91:91+96]
                elif k == 3:
                    target = sketches[:,:, 151:151+96, 85:85+96]
                elif k == 4:
                    target = sketches

                output = model(sketches)[k]
                running_log[key] = output
                loss = loss_fn(output, target)
                running_loss[key] = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if i % log_rate == 0:
        print(f'saving log at: {epoch}.')
        wandb.log({
            "train_loss_eye1": running_loss['eye1'],
            "train_loss_eye2": running_loss['eye2'],
            "train_loss_nose": running_loss['nose'],
            "train_loss_mouth": running_loss['mouth'],
            "train_loss_face": running_loss['face'],
            "train_sketches": [wandb.Image(sketch) for sketch in sketches],
            "train_rec": [wandb.Image(rec) for rec in running_log['face']],
            "train_target_mouth": [wandb.Image(mouth) for mouth in sketches[:,:, 151:151+96, 85:85+96]],
            "train_rec_mouth": [wandb.Image(eye1) for eye1 in running_log['mouth']],
        })
        
        model.eval()
        with torch.no_grad():
            testing_loss = {
                'eye1' : 0,
                'eye2' : 0,
                'nose' : 0,
                'mouth' : 0,
                'face' : 0
            }
            testing_log = {
                'eye1' : [],
                'eye2' : [],
                'nose' : [],
                'mouth' : [],
                'face' : []    
            }
            
            if epoch % args.log_rate == 0:
                print(f'saving model at: epoch {epoch}.')
                torch.save(model.state_dict(), f'{args.log_dir}/{args.date}/{epoch}.pt')
            for i, sketches in enumerate(test_loader):
                if i == 0:
                    sketches = sketches.to(args.device)
                    for k, key in enumerate(encoder_key):
                        if k == 0:
                            target = sketches[:,:, 54:54+64, 78:78+64]
                        elif k == 1:
                            target = sketches[:,:, 128:128+64, 78:78+64]
                        elif k == 2:
                            target = sketches[:,:, 91:91+96, 116:116+96]
                        elif k == 3:
                            target = sketches[:,:, 85:85+96, 151:151+96]
                        elif k == 4:
                            target = sketches

                        output = model(sketches)[k]
                        testing_log[key] = output
                        loss = loss_fn(output, target)
                        testing_loss[key] = loss
                    
                    wandb.log({
                            "test_loss_eye1": testing_loss['eye1'],
                            "test_loss_eye2": testing_loss['eye2'],
                            "test_loss_nose": testing_loss['nose'],
                            "test_loss_mouth": testing_loss['mouth'],
                            "test_loss_face": testing_loss['face'],
                            "test_sketches": [wandb.Image(sketch) for sketch in sketches],
                            "test_rec": [wandb.Image(rec) for rec in testing_log['face']],
                            "test_target_mouth": [wandb.Image(mouth) for mouth in sketches[:,:, 151:151+96, 85:85+96]],
                            "test_rec_mouth": [wandb.Image(eye1) for eye1 in testing_log['mouth']],
                        })



if __name__ == "__main__":
    args = get_args_parser()
    main()