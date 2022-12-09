import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AttModule(nn.Module):
    def __init__(self, featureSize, decodeSize, attSize, dropout=0.5):

        super(AttModule, self).__init__()

        # FC with bottom up features
        self.att_feat = weight_norm(nn.Linear(featureSize, attSize))  
        
        # FC with hidden state values (hidden 1)
        self.att_decoder = weight_norm(nn.Linear(decodeSize, attSize))  
        
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

        # In order to visualize the bounding box attentions, we return the attentions only in this model
        return aw_images, sigmoid


class DecoderAttModule(nn.Module):

    def __init__(self, attSize, embedSize, decodeSize, vocabSize, featureSize=2048, dropout=0.5):

        super(DecoderAttModule, self).__init__()

        # Decleration of the needed sizes
        self.attSize = attSize
        self.decodeSize = decodeSize
        self.vocabSize = vocabSize
        self.featureSize = featureSize
        self.embedSize = embedSize
        self.dropout = dropout

    
        # Attention module to create weighted features
        self.attModule = AttModule(featureSize, decodeSize, attSize) 
    
        self.embedding = nn.Embedding(vocabSize, embedSize)
        
        # Top-down LSTM layer
        self.TD = nn.LSTMCell(embedSize + featureSize + decodeSize, decodeSize, bias=True)
        # Language LSTM layer
        self.lang_layer = nn.LSTMCell(featureSize + decodeSize, decodeSize, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(p=self.dropout)
        
        #FC with hidden state 2 layer to obtain predictions
        self.linear = weight_norm(nn.Linear(decodeSize, vocabSize))

        self.sigmoid = nn.Sigmoid()
        
        self.init_weights() 

    def init_weights(self):

        # Initialization weights uniformly with some small non-zero values.        
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)  

    def init_hidden_state(self, batchSize):

        # Initialization of hidden states of LSTM layers
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
        
        embeddings = self.embedding(sequences) 

        # Initialize LSTM state
        hidden1, cell1 = self.init_hidden_state(batchSize)
        hidden2, cell2 = self.init_hidden_state(batchSize)


        # Prediction score initialization
        preds = torch.zeros(batchSize, max(decode_lengths), vocabSize).to(device)

        for timestep in range(max(decode_lengths)):
            bSize = sum([seq_length > timestep for seq_length in decode_lengths])
            
            # 2)(bSize, decodeSize)
            hidden1,cell1 = self.TD(
                torch.cat([hidden2[:bSize],featsAvg[:bSize],embeddings[:bSize, timestep, :]], dim=1),(hidden1[:bSize], cell1[:bSize]))


            # 1)(bSize, featureSize)
            aw_images, sigmoid = self.attModule(feats[:bSize],hidden1[:bSize])
            
            # 2) (bsize, decoderSize)
            hidden2,cell2 = self.lang_layer(
                torch.cat([aw_images[:bSize],hidden1[:bSize]], dim=1),
                (hidden2[:bSize], cell2[:bSize]))
            
            # 3) (bSize, vocabSize)
            predictions = self.linear(self.dropout(hidden2))
            preds[:bSize, timestep, :] = predictions


        return preds, sequences, decode_lengths, sigmoid, positions






            
