import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class AblationAttModule(nn.Module):
    def __init__(self, featureSize, decodeSize, attSize, dropout=0.5):

        super(AblationAttModule, self).__init__()

        self.relu = nn.ReLU()

        # FC with bottom up features
        self.att_feat = weight_norm(nn.Linear(featureSize, attSize))  
        
        # FC with hidden state values (hidden 1)
        self.att_decoder = weight_norm(nn.Linear(decodeSize, attSize))  
        
        # for softmax
        self.att = weight_norm(nn.Linear(attSize, 1))
        
        # overfitting, classification, and activation. 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, BU_feats, h1):

        # FC layer with hidden states
        h1_att = self.att_decoder(h1)

        # FC layer with bottom up features obtained. 
        img_att = self.att_feat(BU_feats)     
        
        # Attention module forward step
        att1 = self.dropout(self.relu(h1_att.unsqueeze(1) + img_att))
        att2 = self.att(att1)
        attention = att2.squeeze(2)  
        sigmoid = self.softmax(attention)  
        
        # Obtained attended image features (attention weighted image features)
        aw_images = torch.sum((BU_feats * sigmoid.unsqueeze(2)), dim=1) 

        return aw_images


class DecoderAblationAttModule(nn.Module):

    def __init__(self, attSize, embedSize, decodeSize, vocabSize, featureSize=2048, dropout=0.5):
        
        super(DecoderAblationAttModule, self).__init__()

        # Decleration of the needed sizes
        self.attSize = attSize
        self.decodeSize = decodeSize
        self.vocabSize = vocabSize
        self.featureSize = featureSize
        self.embedSize = embedSize
        
        # Attention module to create 
        self.attModule = AblationAttModule(featureSize, decodeSize, attSize) 

        self.embedding = nn.Embedding(vocabSize, embedSize)
        
        # In this model we have used only 1 Layer of LSTM combining images and word embbedings.
        self.LSTM = nn.LSTMCell(embedSize + featureSize, decodeSize, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(p=self.dropout)
        
        #FC with hidden layer to obtain predictions
        self.linear_hidden = weight_norm(nn.Linear(decodeSize, vocabSize))  

        #FC with attention weighted features to obtain predictions
        self.linear_image = weight_norm(nn.Linear(featureSize, vocabSize))
        
        # Overfittting issue
        self.dropout = dropout

        # Initialize weights
        self.init_weights() 

    def init_weights(self):
        
        # Initialization weights uniformly with some small non-zero values.
        self.linear_hidden.bias.data.fill_(0)
        self.linear_hidden.weight.data.uniform_(-0.1, 0.1)
        self.linear_image.bias.data.fill_(0)
        self.linear_image.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)  

    def init_hidden_state(self, batchSize):

        # Initialization of hidden states of LSTM layer
        hidden_states = torch.zeros(batchSize,self.decodeSize).to(device)
        cell_states = torch.zeros(batchSize,self.decodeSize).to(device)

        return hidden_states, cell_states


    def forward(self, feats, sequences, sizes):

        vocabSize = self.vocabSize

        batchSize = feats.size(0)

        # Flatten image
        featsAvg = torch.mean(feats, dim=1).to(device)

        # We sort the data in order to implement the 'timestep' strategy described in the paper (3.2.2)
        sizes, positions = torch.sort((torch.squeeze(sizes, 1)), dim=0, descending=True)

        # We will not encode the end of sentence token <end>.
        decode_lengths = torch.Tensor.tolist((sizes - 1))

        # Sorted parameters
        sequences = sequences[positions]
        feats = feats[positions]
        featsAvg = featsAvg[positions]
        
        embeddings = self.embedding(sequences)  # (batchSize, max_size, embedSize)

        # Initialize LSTM state
        hidden, cell = self.init_hidden_state(batchSize)  # (batchSize, decodeSize)
        
        # Prediction score initialization
        preds = torch.zeros(batchSize, max(decode_lengths), vocabSize).to(device)

        # Sequence generation step considering the max length in the batch size
        for timestep in range(max(decode_lengths)):
            bSize = sum([seq_length > timestep for seq_length in decode_lengths])
            # 1) (bSize, featureSize)
            aw_images = self.attModule(feats[:bSize],hidden[:bSize])
            # 2) (bsize, decoderSize)
            hidden,cell = self.LSTM(torch.cat([aw_images[:bSize], embeddings[:bSize, timestep, :]], dim=1),(hidden[:bSize], cell[:bSize]))  
            # 3) (bSize, vocabSize)
            predictions = self.linear_hidden(self.dropout(hidden)) + self.linear_image(self.dropout(aw_images))

            preds[:bSize, timestep, :] = predictions

        return preds, sequences, decode_lengths, positions










            
