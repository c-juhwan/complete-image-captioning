import os
import sys
import time
import psutil
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.dataset import CocoDataset, collate_fn
from model.captioning_model import CaptioningModel
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name

def training(args:argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Define default transform
    """
    https://pytorch.org/vision/stable/models.html

    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    where H and W are expected to be at least 224. 
    
    The images have to be loaded in to a range of [0, 1] 
    and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
    You can use the following transform to normalize:

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    """
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.image_crop_size), # crop 224x224 from 256x256 image
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406),        # normalize with predefined mean & std above
                             (0.229, 0.224, 0.225))])

    # Load data
    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CocoDataset(args.data_train_image_path, args.data_train_annotation_path, args.spm_model_path,
                                        vocab_size=args.vocab_size, transform=transform)
    dataset_dict['valid'] = CocoDataset(args.data_valid_image_path, args.data_valid_annotation_path, args.spm_model_path,
                                        vocab_size=args.vocab_size, transform=transform)
                                    
    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True,  pin_memory=True, drop_last=True, collate_fn=collate_fn)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False,  pin_memory=True, drop_last=False, collate_fn=collate_fn)
    write_log(logger, "Loaded data")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")

    # Get model instance
    write_log(logger, "Generating model")
    model = CaptioningModel(embed_dim=args.decoder_embed_dim, 
                            encoder_type=args.encoder_type, encoder_pretrained=args.encoder_pretrained,
                            decoder_type=args.decoder_type, hidden_dim=args.decoder_hidden_dim, 
                            nhead=args.decoder_nhead, num_layers=args.decoder_num_layers,
                            bidirectional=args.decoder_bidirectional, 
                            vocab_size=args.vocab_size, max_seq_len=args.trg_max_len)
    model = model.to(device)

    # Get optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloader_dict['train']))
    scaler = GradScaler()

    # Get loss function
    criterion = nn.NLLLoss(ignore_index=args.pad_id)

    # If resume training, load checkpoint
    start_epoch = 0
    if args.training_resume:
        write_log(logger, "Resuming training model")
        checkpoint = torch.load(os.path.join(args.checkpoint_path, f'checkpoint_{args.model_name}.pth.tar'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        write_log(logger, f"Resumed training model from epoch {start_epoch}")
        del checkpoint
    
    # Training
    best_epoch_idx = 0
    best_valid_acc = 0

    for epoch_idx in range(start_epoch, args.num_epochs):
        # Train
        model.train()
        for iter_idx, datas_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), 
                                              desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]')):
            # Train - Get batched data
            images = datas_dicts['images'].to(device, non_blocking=True)
            caption_ids = datas_dicts['caption_ids'].to(device, non_blocking=True)

            # Train - Define target
            targets = caption_ids[:, 1:] # (batch_size, max_seq_len-1), remove <bos>
            #non_pad = targets != args.pad_id
            #targets = targets[non_pad].contiguous().view(-1) # (batch_size*max_seq_len-1)
            targets = targets.contiguous().view(-1) # (batch_size*max_seq_len-1)

            # Train - Forward
            with autocast():
                output_probs = model(images, caption_ids[:, :-1]) # (batch_size, max_seq_length-1, vocab_size), remove <eos>
                output_probs = output_probs.view(-1, output_probs.size(-1)) # (batch_size * max_seq_length-1, vocab_size)

                loss = criterion(output_probs, targets)
            
            # Train - Optimizer with scaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # Train - Scheduler
            scheduler.step()

            # Train - Log to tensorboard
            if args.use_tensorboard:
                accuracy = (output_probs.max(dim=1)[1] == targets).sum() / len(targets)

                writer.add_scalar('TRAIN/Loss', loss.item(),
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)
                writer.add_scalar('TRAIN/Accuracy', accuracy * 100,
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)
                writer.add_scalar('TRAIN/Learning_Rate', optimizer.param_groups[0]['lr'],
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)
                writer.add_scalar('TRAIN/CPU_Usage', psutil.cpu_percent(),
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)
                writer.add_scalar('TRAIN/RAM_Usage', psutil.virtual_memory().percent,
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)
                writer.add_scalar('TRAIN/GPU_Usage', torch.cuda.memory_allocated(device=device),
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)

        # Validation
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']),
                                                   desc=f'Validation - Epoch [{epoch_idx}/{args.num_epochs}]')):
            # Validation - Get batched data
            images = data_dicts['image'].to(device, non_blocking=True)
            caption_ids = data_dicts['caption_id'].to(device, non_blocking=True)
            lengths = data_dicts['length'].to(device, non_blocking=True)

            # Validation - Define target
            targets = caption_ids[:, 1:] # (batch_size, max_seq_len-1), remove <bos>
            non_pad = targets != args.pad_id
            targets = targets[non_pad].contiguous().view(-1) # (batch_size*max_seq_len-1)

            # Validation - Forward
            with torch.no_grad():
                output_probs = model(images, caption_ids[:, :-1], lengths) # (batch_size, max_seq_length-1, vocab_size), remove <eos>
                output_probs = output_probs.view(-1, output_probs.size(-1)) # (batch_size * max_seq_length-1, vocab_size)

                loss = criterion(output_probs, targets)
                accuracy = (output_probs.max(dim=1)[1] == targets).sum() / len(targets)
            
            valid_loss += loss.item()
            valid_accuracy += accuracy.item()

            # Validation - Log to tensorboard
            if args.use_tensorboard:
                writer.add_scalar('VALID/CPU_Usage', psutil.cpu_percent(),
                                epoch_idx * len(dataloader_dict['valid']) + iter_idx)
                writer.add_scalar('VALID/RAM_Usage', psutil.virtual_memory().percent,
                                epoch_idx * len(dataloader_dict['valid']) + iter_idx)
                writer.add_scalar('VALID/GPU_Usage', torch.cuda.memory_allocated(device=device),
                                epoch_idx * len(dataloader_dict['valid']) + iter_idx)

        # Validation - Check accuracy & save checkpoint
        valid_loss /= len(dataloader_dict['valid'])
        valid_accuracy /= len(dataloader_dict['valid'])
        if valid_accuracy > best_valid_acc:
            best_epoch_idx = epoch_idx
            best_valid_acc = valid_accuracy

            save_file_name = os.path.join(args.checkpoint_path,
                                          f'checkpoint_{args.model_name}.pth.tar')
            torch.save({
                        'epoch': epoch_idx,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
            }, save_file_name)

            write_log(logger, f'Best valid at epoch {best_epoch_idx} - {best_valid_acc*100:.4f}%')
            write_log(logger, f'Saved checkpoint to {save_file_name}')
        else:
            write_log(logger, f'Still not better than epoch {best_epoch_idx} - {best_valid_acc*100:.4f}%')


        # Validation - Log to tensorboard
        if args.use_tensorboard:
            writer.add_scalar('VALID/Loss', valid_loss.item(),
                              epoch_idx)
            writer.add_scalar('VALID/Accuracy', valid_accuracy * 100,
                              epoch_idx)
    
    # Done!
    write_log(logger, f'Done! Best epoch: {best_epoch_idx} - {best_valid_acc*100:.4f}%')

    # Save final model
    save_file_name = os.path.join(args.model_path, 
                                  f'{args.model_name}_final.pth.tar')
    torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
    }, save_file_name)
    write_log(logger, f'Saved final model to {save_file_name}')