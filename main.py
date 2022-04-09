import time
import argparse

from task.preprocessing import preprocessing
from task.train import training
from task.test import testing
from utils import check_path, set_random_seed

def main(args):
    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    # Check if the path exists
    check_path(args.data_path)
    check_path(args.checkpoint_path)
    check_path(args.model_path)
    check_path(args.tensorboard_path)

    if args.preprocessing:
        preprocessing(args)
    elif args.training:
        training(args)
    elif args.testing:
        testing(args)
    else:
        raise ValueError('Please specify the task to run.')
    
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument('--preprocessing', action='store_true',
                        help='Run preprocessing.')
    parser.add_argument('--training', action='store_true',
                        help='Run training.')
    parser.add_argument('--training_resume', action='store_true',
                        help='Resume training.')
    parser.add_argument('--testing', action='store_true',
                        help='Run testing.')

    # Path
    parser.add_argument('--preprocess_path', default='./preprocessing/', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', type=str, default='./dataset/',
                        help='Path to the data folder.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/',
                        help='Path to the checkpoint folder.')
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='Path to the result model folder.')
    
    # Preprocessing - Annotation path
    parser.add_argument('--data_train_annotation_path', type=str, default='./dataset/annotations/captions_train2017.json',
                        help='Path to the train annotations file')
    parser.add_argument('--data_valid_annotation_path', type=str, default='./dataset/annotations/captions_val2017.json',
                        help='Path to the validation annotations file')
    # Preprocessing - Image path
    parser.add_argument('--data_train_image_path', type=str, default='./dataset/train2017/',
                        help='Path to the train images folder')
    parser.add_argument('--data_valid_image_path', type=str, default='./dataset/val2017/',
                        help='Path to the validation images folder')
    parser.add_argument('--data_test_image_path', type=str, default='./dataset/test2017/',
                        help='Path to the validation images folder')
    # Preprocessing - Caption txt path
    parser.add_argument('--data_train_caption_path', type=str, default='./preprocessing/train_captions.txt',
                        help='Path to the train captions file after preprocessing')
    parser.add_argument('--data_valid_caption_path', type=str, default='./preprocessing/valid_captions.txt',
                        help='Path to the validation captions file after preprocessing')
    # Preprocessing - Spm model path
    parser.add_argument('--spm_model_path', type=str, default='./preprocessing/',
                        help='Path to the sentencepiece model')
    # Preprocessing - Spm config
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
    parser.add_argument('--trg_max_len', default=20, type=int,
                        help='Maximum length of target sequence; Default is 20')
    # Preprocessing - Image preprocessing config
    parser.add_argument('--resize_image_size', type=int, default=256,
                        help='Size of resized image after preprocessing.')
    parser.add_argument('--resize_image_path', type=str, default='./preprocessing/',
                        help='Path to the resized images folder.')

    # Model Config
    parser.add_argument('--model_name', default="CaptioningModel", type=str,
                        help='Name of the model; Default is CaptioningModel')
    # Model - Encoder Parameters
    parser.add_argument('--encoder_type', default='resnet152', type=str,
                        help='Encoder type; Default is resnet152')
    parser.add_argument('--encoder_pretrained', default=True, type=bool,
                        help='Whether to use pretrained encoder; Default is True')
    # Model - Decoder Parameters
    parser.add_argument('--decoder_type', default='gru', type=str,
                        help='Decoder type; Default is gru')
    parser.add_argument('--decoder_embed_dim', default=256, type=int,
                        help='Decoder embedding dimension == encoder output dimension; Default is 256')
    parser.add_argument('--decoder_hidden_dim', default=512, type=int,
                        help='Decoder hidden dimension; If decoder_type==transformer, then d_model; Default is 512')
    parser.add_argument('--decoder_nhead', default=8, type=int,
                        help='Decoder nhead for decoder_type==transformer; Default is 8')
    parser.add_argument('--decoder_num_layers', default=1, type=int,
                        help='Decoder number of layers; If decoder_type==transformer, then recommend value is 6; Default is 1')
    parser.add_argument('--decoder_bidirectional', default=False, type=bool,
                        help='Decoder bidirectional; Default is False')

    # Optimizer & Scheduler
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD'; Default is AdamW")
    parser.add_argument('--scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")

    # Training
    parser.add_argument('--num_epochs', default=10, type=int, 
                        help='Training epochs; Default is 10')
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='Num CPU Workers; Default is 2')
    parser.add_argument('--batch_size', default=16, type=int,    
                        help='Batch size; Default is 16')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Image crop size; Default is 224')

    # Testing
    parser.add_argument('--test_batch_size', default=32, type=int, 
                        help='Test batch size; Default is 32')
    parser.add_argument('--beam_size', default=5, type=int, 
                        help='Beam search size; Default is 5')
    parser.add_argument('--beam_alpha', default=0.7, type=float, 
                        help='Beam search length normalization; Default is 0.7')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, 
                        help='Beam search repetition penalty term; Default is 1.3')

    # Seed & Logging
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed; Default is 42')
    parser.add_argument('--use_tensorboard', default=True, type=bool,
                        help='Using tensorboard; Default is True')
    parser.add_argument('--tensorboard_path', default='./tensorboard_runs', type=str,
                        help='Tensorboard log path; Default is ./tensorboard_runs')
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')

    args = parser.parse_args()
    main(args)