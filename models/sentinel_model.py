import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Adap_AttModule(nn.Module):

    def __init__(self, featureSize, decodeSize, attSize, dropout=0.5):
        
        super(Adap_AttModule, self).__init__()


        # FC with bottom up features
        self.att_feat = weight_norm(nn.Linear(featureSize, attSize))  
    
        # FC with hidden state values (hidden 1)
        self.att_decoder = weight_norm(nn.Linear(decodeSize, attSize))  
        
        # FC with sentinel vector (hidden 1)
        self.att_sent = weight_norm(nn.Linear(decodeSize, attSize))
        self.weight_sent = weight_norm(nn.Linear(decodeSize, featureSize))
    

        self.att = weight_norm(nn.Linear(attSize, 1))


        # overfitting, classification, and activation.
        self.softmax = nn.Softmax(dim=1)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, BU_feats, h1, sent):
        

        # FC layer with hidden states
        h1_att = self.att_decoder(h1)

        # FC later with bottom up features obtained
        img_att = self.att_feat(BU_feats)  

        # Attention for sentinel vector which will be concatenated with BU attention
        sent_att = self.att_sent(sent)
        
        # Attention module for bottum up features
        att1 = self.dropout(self.relu(h1_att.unsqueeze(1) + img_att))
        att2 = self.att(att1)
        attention_img = att2.squeeze(2)

        # Attention module for sentinel vector
        sentinel_att1 = self.dropout(self.relu(sent_att + h1_att))
        sentinel_att = self.att(sentinel_att1)
        
        # Concatenate the attentions
        attention = torch.cat([attention_img, sentinel_att], dim=1)
        
        # Attention weighted sentinel vector
        sigmoid = self.softmax(attention)  
        aw_sentinel = sent * sigmoid[:, -1].unsqueeze(1)
        aw_sentinel = self.weight_sent(self.dropout(aw_sentinel)) 

        # Attention weighted images ( attention weighted BU features + attention weighted sentinel)
        aw_images = torch.sum((BU_feats * sigmoid[:, :-1].unsqueeze(2)), dim=1) + aw_sentinel
        return aw_images


class DecoderAdap_AttModule(nn.Module):
    

    def __init__(self, attSize, embedSize, decodeSize, vocabSize, featureSize=2048, dropout=0.5):
        
        super(DecoderAdap_AttModule, self).__init__()
        # Decleration of the needed sizes
        self.attSize = attSize
        self.decodeSize = decodeSize
        self.vocabSize = vocabSize
        self.featureSize = featureSize
        self.embedSize = embedSize
        self.dropout = dropout

        # Attention module to create weighted features
        self.adap_attModule = Adap_AttModule(featureSize, decodeSize, attSize) 
    
        self.embedding = nn.Embedding(vocabSize, embedSize)
        
        # Top-down LSTM layer
        self.TD = nn.LSTMCell(embedSize + featureSize + decodeSize, decodeSize, bias=True)

        # Language LSTM layer
        self.lang_layer = nn.LSTMCell(featureSize + decodeSize, decodeSize, bias=True)
        
        # FC layers for gate to compute sentinel vector based on memory cell 
        self.gate_hidden_weight = weight_norm(nn.Linear(decodeSize, decodeSize))
        self.gate_embedding_weight = weight_norm(nn.Linear(embedSize, decodeSize))

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout)
        
        #FC with hidden state 2 layer to obtain predictions
        self.linear_hidden = weight_norm(nn.Linear(decodeSize, vocabSize))

        #FC with attention weighted features to obtain predictions
        self.linear_image = weight_norm(nn.Linear(featureSize, vocabSize))
 
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights() 

    def init_weights(self):

        # Initialization weights uniformly with some small non-zero values.
        self.linear_hidden.bias.data.fill_(0)
        self.linear_hidden.weight.data.uniform_(-0.1, 0.1)
        self.linear_image.bias.data.fill_(0)
        self.linear_image.weight.data.uniform_(-0.1, 0.1)
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
        feats = feats[positions]
        featsAvg = featsAvg[positions]
        sequences = sequences[positions]

        # Embedding
        embeddings = self.embedding(sequences)  

        # Initialize LSTM state
        hidden1, cell1 = self.init_hidden_state(batchSize)  
        hidden2, cell2 = self.init_hidden_state(batchSize) 

        # Prediction score initialization
        preds = torch.zeros(batchSize, max(decode_lengths), vocabSize).to(device)

        for timestep in range(max(decode_lengths)):
            bSize = sum([seq_length > timestep for seq_length in decode_lengths])

            # hidden state and cell state obtained from top-down lstm layer
            hidden1, cell1 = self.TD(
                torch.cat([hidden2[:bSize],featsAvg[:bSize],embeddings[:bSize, timestep, :]], dim=1),(hidden1[:bSize], cell1[:bSize]))
            

            # gate to compute sentinel vector based on memory cell (depends on embeddings (words) and first hidden state)
            gate = self.sigmoid(self.gate_embedding_weight(self.dropout(embeddings[:bSize, timestep, :]))
                                   + self.gate_hidden_weight(self.dropout(hidden1[:bSize])))    

            # Determine to attend images or not based on gate vector
            sent = gate * torch.tanh(cell1[:bSize])

            # Attention weighted image features (BU features + sentinel)
            aw_images = self.adap_attModule(feats[:bSize], hidden1[:bSize], sent)

            # hidden state and cell state obtained from language lstm layer
            hidden2,cell2 = self.lang_layer(
                torch.cat([aw_images[:bSize],hidden1[:bSize]], dim=1),
                (hidden2[:bSize], cell2[:bSize]))

            predictions = self.linear_hidden(self.dropout(hidden2)) + self.linear_image(self.dropout(aw_images))

            preds[:bSize, timestep, :] = predictions

        return preds, sequences, decode_lengths, positions








            


