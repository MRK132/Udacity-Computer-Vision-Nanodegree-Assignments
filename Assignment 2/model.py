import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embed(captions)
        embed = torch.cat((features.unsqueeze(1), embed), 1)
        lstm_output, _ = self.lstm(embed)
        output = self.fc(lstm_output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            output_lstm = outputs.squeeze(1)
            output = self.fc(output_lstm)
            
            final_word = output.max(1)[1]
            predicted_sentence.append(final_word.item())
            inputs = self.embed(final_word).unsqueeze(1)
            
        return predicted_sentence
            