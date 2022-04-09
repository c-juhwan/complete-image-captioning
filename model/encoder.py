import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, output_dim:int, encoder_type:str='resnet152', encoder_pretrained:bool=True):
        """
        Encoder of the model.
        Args:
            output_dim (int): The output dimension of the encoder, 
                              which should be the same as the embedding dimension of the decoder.
            encoder_type (str): The type of encoder to use.
            encoder_pretrained (bool): Whether to use pretrained encoder.
        """
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type.lower()

        # Define the pretrained decoder
        if self.encoder_type == 'resnet152':
            resnet = models.resnet152(pretrained=encoder_pretrained)
            modules = list(resnet.children())[:-1]
            self.extractor_out_dim = resnet.fc.in_features
            self.extractor = nn.Sequential(*modules)
        elif self.encoder_type == 'efficientnet_b7':
            effnet = models.efficientnet_b7(pretrained=encoder_pretrained)
            modules = list(effnet.children())[:-1]
            self.extractor_out_dim = effnet.classifier[1].in_features
            self.extractor = nn.Sequential(*modules)
        elif self.encoder_type == 'vit_b_16':
            vit = models.vit_b_16(pretrained=encoder_pretrained)
            modules = list(vit.children())[:-1]
            self.extractor_out_dim = vit.heads[0].in_features
            self.extractor = nn.Sequential(*modules)
        else:
            raise NotImplementedError(f'Encoder type {encoder_type} is not implemented')

        # Define the output layer
        self.encoder_linear1 = nn.Linear(self.extractor_out_dim, self.extractor_out_dim // 2)
        self.activation = nn.ReLU()
        self.encoder_linear2 = nn.Linear(self.extractor_out_dim // 2, output_dim)
        self.out = nn.Sequential(self.encoder_linear1, self.activation, self.encoder_linear2)

    def forward(self, images:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extractor(images)
        features = features.view(features.size(0), -1) # Flatten to (batch_size, features_dim)
        features = self.out(features) # (batch_size, output_dim)

        return features