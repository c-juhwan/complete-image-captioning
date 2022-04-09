import os
import pickle
import argparse
import sentencepiece as spm
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from PIL import Image

def save_annotations_to_txt(annotation_path:str, caption_path:str):
    """
    Load annotations file from COCO and save captions to txt file.
    """
    coco = COCO(annotation_path)
    ids = coco.anns.keys()

    captions = []
    for i, id in enumerate(tqdm(ids, total=len(ids), desc="Loading annotations from %s" % annotation_path)):
        caption = str(coco.anns[id]['caption'])
        captions.append(caption)
    
    with open(caption_path, 'w') as f:
        for caption in tqdm(captions, total=len(captions), desc="Saving captions to %s" % caption_path):
            f.write(caption + '\n')

def train_spm_model(args:argparse.Namespace):
    """
    Train sentencepiece model.
    """
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.data_train_caption_path} --model_prefix={args.spm_model_path}spm_model_{args.vocab_size} '
        f'--vocab_size={args.vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}'
    )

def tokenize_sentences(args:argparse.Namespace):
    # Define
    processed_trg, word2id = {}, {}
    processed_trg['train'], processed_trg['valid'] = {}, {}

    # Load sentencepiece model
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{args.spm_model_path}/spm_model_{args.vocab_size}.model')

    # Make word2id dictionary
    trg_vocab = []
    with open(f'{args.spm_model_path}/spm_model_{args.vocab_size}.vocab', 'r') as f:
        for line in f:
            trg_vocab.append(line[:-1].split('\t')[0])
    word2id['trg'] = {w: i for i, w in enumerate(trg_vocab)}

    # Load files
    trg_sequences = {}
    with open(args.data_train_caption_path, 'r') as f:
        trg_sequences['train'] = [x.replace('\n', '') for x in f.readlines()]
    with open(args.data_valid_caption_path, 'r') as f:
        trg_sequences['valid'] = [x.replace('\n', '') for x in f.readlines()]
    
    # Preprocess
    processed_trg['train']['input_ids'] = tuple(
        [args.bos_id] + sp.Encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in trg_sequences['train']
    )
    processed_trg['valid']['input_ids'] = tuple(
        [args.bos_id] + sp.Encode(text, out_type=int) + [args.eos_id] for text in trg_sequences['valid']
    )

    # Attention mask
    processed_trg['train']['attention_mask'] = []
    for ind in processed_trg['train']['input_ids']:
        processed_trg['train']['attention_mask'].append([1 if i <= ind.index(args.eos_id) else 0 for i in range(args.trg_max_len)])

    processed_trg['valid']['attention_mask'] = []
    for ind in processed_trg['valid']['input_ids']:
        processed_trg['valid']['attention_mask'].append([1 if i <= ind.index(args.eos_id) else 0 for i in range(args.trg_max_len)])
    
    return processed_trg, word2id

def save_data(args:argparse.Namespace, processed_trg:dict, word2id:dict):
    """
    Save data to .pkl file.
    """
    save_path = os.path.join(args.spm_model_path, f'processed_coco_spm_{args.vocab_size}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'train_trg_indices': processed_trg['train']['input_ids'],
            'valid_trg_indices': processed_trg['valid']['input_ids'],
            'train_trg_attention_mask': processed_trg['train']['attention_mask'],
            'valid_trg_attention_mask': processed_trg['valid']['attention_mask'],
            'trg_word2id': word2id['trg']
        }, f)
    
    print(f'Saved processed text data to {save_path}')

def resize_images(args:argparse.Namespace):
    """
    Resize images to given square size.
    """
    splits = ['train', 'valid', 'test']
    image_paths = {}
    image_paths['train'] = args.data_train_image_path
    image_paths['valid'] = args.data_valid_image_path
    image_paths['test'] = args.data_test_image_path

    output_paths = {}
    for split in splits:
        output_paths[split] = os.path.join(args.resize_image_path, split +'/')
        if not os.path.exists(output_paths[split]):
            os.mkdir(output_paths[split])

    images = {}
    for split in splits:
        images[split] = os.listdir(image_paths[split])

    for split in splits:
        for i, image in enumerate(tqdm(images[split], total=len(images[split]), desc="Resizing images from %s" % image_paths[split])):
            with open(os.path.join(image_paths[split], image), 'rb') as f:
                img = Image.open(f)
                img = img.resize((args.resize_image_size, args.resize_image_size), Image.ANTIALIAS)
                img.save(os.path.join(output_paths[split], image), img.format)

def preprocessing(args:argparse.Namespace):
    # Text preprocessing
    save_annotations_to_txt(args.data_train_annotation_path, args.data_train_caption_path)
    save_annotations_to_txt(args.data_valid_annotation_path, args.data_valid_caption_path)
    train_spm_model(args)
    #processed_trg, word2id = tokenize_sentences(args)
    #save_data(args, processed_trg, word2id)

    # Image preprocessing
    resize_images(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Annotation path
    parser.add_argument('--data_train_annotation_path', type=str, default='./dataset/annotations/captions_train2017.json',
                        help='Path to the train annotations file')
    parser.add_argument('--data_valid_annotation_path', type=str, default='./dataset/annotations/captions_val2017.json',
                        help='Path to the validation annotations file')
    
    # Image path
    parser.add_argument('--data_train_image_path', type=str, default='./dataset/train2017/',
                        help='Path to the train images folder')
    parser.add_argument('--data_valid_image_path', type=str, default='./dataset/val2017/',
                        help='Path to the validation images folder')
    parser.add_argument('--data_test_image_path', type=str, default='./dataset/test2017/',
                        help='Path to the validation images folder')

    # Caption txt path
    parser.add_argument('--data_train_caption_path', type=str, default='./preprocessing/train_captions.txt',
                        help='Path to the train captions file after preprocessing')
    parser.add_argument('--data_valid_caption_path', type=str, default='./preprocessing/valid_captions.txt',
                        help='Path to the validation captions file after preprocessing')

    # Spm model path
    parser.add_argument('--spm_model_path', type=str, default='./preprocessing/',
                        help='Path to the sentencepiece model')
    # Spm config
    parser.add_argument('--sentencepiece_model', default='unigram', choices=['unigram', 'bpe', 'word', 'char'],
                        help="SentencePiece model type; Default is unigram")
    parser.add_argument('--vocab_size', default=8000, type=int,
                        help='Source text vocabulary size; Default is 8000')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=1, type=int,
                        help='Padding token index; Default is 1')
    parser.add_argument('--eos_id', default=2, type=int,
                        help='Padding token index; Default is 2')
    parser.add_argument('--trg_max_len', default=300, type=int,
                        help='Maximum length of target sequence; Default is 300')
    
    # Image preprocessing config
    parser.add_argument('--resize_image_size', type=int, default=256,
                        help='Size of resized image after preprocessing.')
    parser.add_argument('--resize_image_path', type=str, default='./preprocessing/',
                        help='Path to the resized images folder.')

    args = parser.parse_args()
    preprocessing(args)