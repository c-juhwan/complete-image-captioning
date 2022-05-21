import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import sentencepiece as spm
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    def __init__(self, image_dir_path:str, annotation_path:str, spm_model_path:str,
                 vocab_size:int=8000, transform=None, min_seq_len:int=4, max_seq_len:int=20):
        self.image_dir_path = image_dir_path
        self.coco = COCO(annotation_path)
        self.ids = list(self.coco.anns.keys())
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f'{spm_model_path}/spm_model_{vocab_size}.model')
        
        self.transform = transform
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
    
    def __getitem__(self, index:int):
        annotation_id = self.ids[index]
        caption = self.coco.anns[annotation_id]['caption']
        image_id = self.coco.anns[annotation_id]['image_id']
        image_path = self.coco.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.image_dir_path, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert caption to word ids & add <bos> and <eos>
        caption_id = self.sp.EncodeAsIds(caption)
        caption_id = torch.Tensor(caption_id)

        # Get valid length of caption, without padding
        length = len(caption_id)
        length = torch.Tensor([length]).long()

        # Pad caption to max_seq_len
        caption_tensor = torch.zeros(self.max_seq_len).long()
        caption_tensor[0] = torch.Tensor([self.sp.bos_id()])
        caption_tensor[-1] = torch.Tensor([self.sp.eos_id()])
        if length < self.max_seq_len:
            caption_tensor[1:length+1] = caption_id
        else: # If caption is longer than max_seq_len, truncate it
            caption_tensor[1:] = caption_id[:self.max_seq_len-1]

        assert caption_tensor.shape[0] == self.max_seq_len

        data_dict = {
            'image': image,
            'caption_id': caption_tensor,
            'length': length
        }
        return data_dict
    
    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """
    Create mini-batch tensors from the list of dictionaries.
    :param data: list of dictionary containing data
    :return: mini-batch tensors
    """
    data.sort(key=lambda x: x['length'], reverse=True)
    images = torch.stack([d['image'] for d in data], dim=0) # (batch_size, 3, 224, 224)
    caption_ids = torch.stack([d['caption_id'] for d in data], dim=0) # (batch_size, max_seq_len)
    lengths = torch.stack([d['length'] for d in data], dim=0) # (batch_size)

    datas_dict = {
        'images': images,
        'caption_ids': caption_ids,
        'lengths': lengths
    }
    return datas_dict