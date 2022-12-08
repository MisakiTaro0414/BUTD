import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Saliency_AttModule(nn.Module):

    def __init__(self, featureSize, saliencySize, decodeSize, attSize, dropout=0.5):
        
        super(Saliency_AttModule, self).__init__()


        # FC with bottom up features to compute attention
        self.att_feat = weight_norm(nn.Linear(featureSize, attSize))  
    
        # FC with hidden state values (hidden 1) to compute attention
        self.att_decoder = weight_norm(nn.Linear(decodeSize, attSize))  

        # FC with saliency to match dimension
        self.weight_saliency = weight_norm(nn.Linear(saliencySize, featureSize))
        
        # FC with saliency features to compute attention
        self.att_saliency = weight_norm(nn.Linear(featureSize, attSize))


        self.att = weight_norm(nn.Linear(attSize, 1))


        # overfitting, classification, and activation.
        self.softmax = nn.Softmax(dim=1)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, BU_feats, saliency_feats, h1):
        

        # FC layer with hidden states
        h1_att = self.att_decoder(h1)

        # FC later with bottom up features obtained
        img_att = self.att_feat(BU_feats)  

        # FC layer to match dimension with BU_feats:
        saliency_feats = self.weight_saliency(saliency_feats)

        # Attention for Saliency features which will be concatenated with BU attention
        saliency_att = self.att_saliency(saliency_feats)
        
        # Attention module for bottum up features
        att1 = self.dropout(self.relu(h1_att.unsqueeze(1) + img_att))
        att2 = self.att(att1)
        attention_img = att2.squeeze(2)

        # Attention module for salienct features
        saliency_att1 = self.dropout(self.relu(saliency_att + h1_att))
        saliency_att = self.att(saliency_att1)
        
        # Concatenate the attentions
        attention = torch.cat([attention_img, saliency_att], dim=1)
        
        sigmoid = self.softmax(attention)  

        # Attention weighted saliency features 
        aw_saliency = saliency_feats * sigmoid[:, -1].unsqueeze(1)
        

        # Attention weighted images ( attention weighted BU features + attention weighted saliency features)
        aw_images = torch.sum((BU_feats * sigmoid[:, :-1].unsqueeze(2)), dim=1) +  aw_saliency
        return aw_images


class DecoderSaliency_AttModule(nn.Module):

    def __init__(self, attSize, embedSize, decodeSize, vocabSize, featureSize=2048, saliencySize=64*64, dropout=0.5):

        super(DecoderSaliency_AttModule, self).__init__()

        # Decleration of the needed sizes
        self.attSize = attSize
        self.decodeSize = decodeSize
        self.vocabSize = vocabSize
        self.featureSize = featureSize
        self.embedSize = embedSize
        self.dropout = dropout

    
        # Attention module to create weighted features
        self.attModule = Saliency_AttModule(featureSize, saliencySize, decodeSize, attSize) 
    
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


    def forward(self, feats, salfeats, sequences, sizes):
    
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
        salfeats = salfeats[positions]

        
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
            aw_images = self.attModule(feats[:bSize], salfeats[:bSize], hidden1[:bSize])
            
            # 2) (bsize, decoderSize)
            hidden2,cell2 = self.lang_layer(
                torch.cat([aw_images[:bSize],hidden1[:bSize]], dim=1),
                (hidden2[:bSize], cell2[:bSize]))
            
            # 3) (bSize, vocabSize)
            predictions = self.linear(self.dropout(hidden2))
            preds[:bSize, timestep, :] = predictions


        return preds, sequences, decode_lengths, positions

    

   