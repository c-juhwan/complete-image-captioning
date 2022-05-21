clear

python main.py --training --encoder_type='resnet152' --decoder_type='rnn'
#python main.py --training --encoder_type='resnet152' --decoder_type='lstm'
#python main.py --training --encoder_type='resnet152' --decoder_type='gru'

#python main.py --training --encoder_type='resnet152' --decoder_type='rnn' --decoder_bidirectional=True
#python main.py --training --encoder_type='resnet152' --decoder_type='lstm' --decoder_bidirectional=True
#python main.py --training --encoder_type='resnet152' --decoder_type='gru' --decoder_bidirectional=True