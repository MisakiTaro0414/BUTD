import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
#from models.sentinel_model import  DecoderAdap_AttModule
#from models.arnet_model import DecoderAttModule
#from models.model import DecoderAttModule
#from models.simplified_model import DecoderAblationAttModule
#from models.arnet_sentinel_model import DecoderARnetAdap_AttModule
from models.saliency_model import DecoderSaliency_AttModule


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


data_root = 'final_dataset' 
data_name = 'coco_5_cap_per_img_5_min_word_freq'


batch_size = 100
embSize = 1024 
decoderSize = 1024  
attentionSize = 1024 
dropout = 0.5
 


continue_epoch = 0 # used in case of resuming the training
bad_epochs = 0  # used for detecting early stopping or learning rate scheduler
epochs = 50  
best_bleu4_score = 0  # BLEU-4 score right now
checkpoint = "results/Saliency_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"  # path to checkpoint


def main():
    
    global best_bleu4_score, bad_epochs, checkpoint, continue_epoch, data_name, mapping

    mapping_file = os.path.join(data_root, 'WORDMAP_' + data_name + '.json')
    with open(mapping_file, 'r') as j:
        mapping = json.load(j)


    if checkpoint is None:
        # Specify the name of selected decoder
        decoder = DecoderSaliency_AttModule(int(attentionSize), int(embSize), int(decoderSize), len(mapping), featureSize=2048, dropout=0.5)
        optimizer = torch.optim.Adamax(params=filter(lambda x: x.requires_grad, decoder.parameters()))

    else:
        checkpoint = torch.load(checkpoint)
        continue_epoch = checkpoint['epoch'] + 1
        bad_epochs = checkpoint['bad_epochs']
        best_bleu4_score = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        optimizer = checkpoint['decoder_optimizer']

    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    
    
    #train_loader = torch.utils.data.DataLoader(CustomDataset(data_root, data_name, 'TRAIN'), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(CustomDataset(data_root, data_name, 'VAL'), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # In case of training with saliency map predictions, please comment the two lines above and uncomment the 2 lines below
    train_loader = torch.utils.data.DataLoader(SaliencyCustomDataset(data_root, data_name, 'TRAIN'), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(SaliencyCustomDataset(data_root, data_name, 'VAL'), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    
    # Learning rate scheduler and Early stopping 
    for epoch in range(continue_epoch, epochs):
        # Decay learning rate if there is no improvement for 5 consecutive epochs, 
        #and terminate training after 20 consequetive non-improvement epochs 
        
        if bad_epochs > 0 and bad_epochs % 5 == 0:
            for parameters in optimizer.param_groups:
                parameters['lr'] = parameters['lr'] * 0.75
    
        if bad_epochs == 20:
            break
    
    
        train(train_loader, decoder, criterion, optimizer, epoch)
        bleu4_score = validate(val_loader, decoder, criterion)

        best = False
        if bleu4_score > best_bleu4_score:
            best_bleu4_score = bleu4_score
            bad_epochs = 0
            best = True
            
        else:
            bad_epochs += 1
            print("Number of epochs after last improvement %d\n" % (bad_epochs,))

        # Save checkpoint according to your model (uncomment the desired one and comment out the rest):

        #save_checkpoint(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score, best)
        #save_checkpoint_sentinel(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score, best)
        #save_checkpoint_arnet(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score, best)
        #save_checkpoint_arnet_sentinel(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score, best)
        save_checkpoint_saliency(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score, best)
        #save_checkpoint_simplified(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score, best):



def train(train_loader, decoder, criterion, optimizer, epoch):
    

    decoder.train()  # Activate Dropout and BatchNormalization

    losses = 0 # loss (per word decoded)
    loss_count = 0
    top5accs = 0  # top5 accuracy
    top5counts = 0

    # Batches
    #for i, (imagefeatures, sequence, sequencelength) in enumerate(train_loader):
    # In case of training with saliency map predictions, please comment the the line above and uncomment the line below as train_loder returns
    # features obtained from saliency as well. 
    for i, (imagefeatures, salfeatures, sequence, sequencelength) in enumerate(train_loader):
    

        imagefeatures = imagefeatures.to(device)
        # Uncomment below in case of saliency model
        salfeatures = salfeatures.to(device)
        sequence = sequence.to(device)
        sequencelength = sequencelength.to(device)

        # Base model (decoder returns attentions in order to visualize the attention on bounding boxes in inference):
        #preds, sorted_sequence, decode_lengths, _, sort_indexes  = decoder(imagefeatures, sequence, sequencelength)

        # Saliency model (decoder takes salfeatures for the forwards pass):
        preds, sorted_sequence, decode_lengths, sort_indexes = decoder(imagefeatures, salfeatures, sequence, sequencelength)

        # ARNet models (decoder returns the ARNet network loss (regularizing loss) ):
        #preds, sorted_sequence, decode_lengths, sort_indexes, loss_ar  = decoder(imagefeatures, sequence, sequencelength)
        
        # Sentinel and Simplified model:
        #preds, sorted_sequence, decode_lengths, sort_indexes = decoder(imagefeatures, sequence, sequencelength)

        
        # skip the <start> token
        labels = sorted_sequence[:, 1:]

        # Use pack padding in any case to have same dimension of preds and labels
        preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
        labels = pack_padded_sequence(labels, decode_lengths, batch_first=True).data

        # Calculate loss (In case of models containing ARNet network, please add loss_ar as well)
        loss = criterion(preds, labels) #+ loss_ar
        
        optimizer.zero_grad()
        loss.backward()

        # Solve vanishing/explosion gradient issue by gradient clipping
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)


        optimizer.step()

        #Calculate top 5 accuracy
        _, indices = preds.topk(5, 1, True, True)
        correct_preds = indices.eq(labels.view(-1, 1).expand_as(indices))
        total_correct_preds = correct_preds.view(-1).float().sum()  
        top5 = total_correct_preds.item() * (100.0 / labels.size(0))
        top5accs += top5
        top5counts += sum(decode_lengths)
        top5_ave = top5accs / top5counts
        losses += loss
        loss_count += sum(decode_lengths)
        loss_ave = losses/loss_count
        

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f} ({loss_ave:.4f})\t'
                  'Top-5 Accuracy {top5:.3f} ({top5_ave:.4f})'.format(epoch, i, len(train_loader),
                                                                          loss=loss, loss_ave =loss_ave, 
                                                                          top5 = top5, top5_ave = top5_ave))


def validate(val_loader, decoder, criterion):
    
    decoder.eval()  # eval mode (no dropout or batchnorm)

    losses = 0
    loss_count = 0
    top5accs = 0
    top5counts = 0


    groundtruths = list()  
    predictions = list()  

    with torch.no_grad(): 
        #for i, (imagefeatures, sequence, sequencelength, sequences_generated) in enumerate(val_loader):
        # In case of training with saliency map predictions, please comment the the line above and uncomment the line below as train_loder returns
        # features obtained from saliency as well. 
        for i, (imagefeatures, salfeatures, sequence, sequencelength, sequences_generated) in enumerate(val_loader):

           
            imagefeatures = imagefeatures.to(device)
            # Uncomment below in case of saliency model
            salfeatures = salfeatures.to(device)
            sequence = sequence.to(device)
            sequencelength = sequencelength.to(device)
            sequences_generated = sequences_generated.to(device)
            


            
            # Base model (decoder returns attentions in order to visualize the attention on bounding boxes in inference):
            #preds, sorted_sequence, decode_lengths, _, sort_indexes  = decoder(imagefeatures, sequence, sequencelength)

            # Saliency model (decoder takes salfeatures for the forwards pass):
            preds, sorted_sequence, decode_lengths, sort_indexes = decoder(imagefeatures, salfeatures, sequence, sequencelength)

            # ARNet models (decoder returns the ARNet network loss (regularizing loss) ):
            #preds, sorted_sequence, decode_lengths, sort_indexes, loss_ar  = decoder(imagefeatures, sequence, sequencelength)
            
            # Sentinel and Simplified model:
            #preds, sorted_sequence, decode_lengths, sort_indexes = decoder(imagefeatures, sequence, sequencelength)

            sequences_generated = sequences_generated[sort_indexes]  # because images were sorted in the decoder

            # skip the <start> token
            labels =sorted_sequence[:, 1:]
            

            copy_preds = preds.clone()
             # Use pack padding in any case to have same dimension of preds and labels
            preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
            labels = pack_padded_sequence(labels, decode_lengths, batch_first=True).data

            # Calculate loss (In case of models containing ARNet network, please add loss_ar as well)            
            loss = criterion(preds, labels) # + loss_ar

            # Top-k accuracy
            _, indices = preds.topk(5, 1, True, True)
            correct_preds = indices.eq(labels.view(-1, 1).expand_as(indices))
            total_correct_preds = correct_preds.view(-1).float().sum()  
            top5 = total_correct_preds.item() * (100.0 / labels.size(0))          
            top5accs += top5
            top5counts += sum(decode_lengths)
            top5_ave = top5accs / top5counts
            losses += loss
            loss_count += sum(decode_lengths)
            loss_ave = losses/loss_count
            

            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss:.4f} ({loss_ave:.4f})\t'
                      'Top-5 Accuracy {top5:.3f} ({top5_ave:.3f})\t'.format(i, len(val_loader), 
                                                                               loss = loss, loss_ave = loss_ave, top5 = top5, top5_ave = top5_ave))

            for i in range(sequences_generated.shape[0]):
                cap = sequences_generated[i].tolist()
                caps = list(map(lambda x: [word for word in x if word not in {mapping['<start>'], mapping['<pad>']}], cap))
                groundtruths.append(caps)

            _, best_preds = torch.max(copy_preds, dim=2)
            best_preds = best_preds.tolist()
            temp = []
            for i, prediction in enumerate(best_preds):
                temp.append(prediction[:decode_lengths[i]])
            best_preds = temp
            predictions.extend(best_preds)




    bleu4_score = round(corpus_bleu(groundtruths, predictions), 4)

    print('\nLoss: {loss_ave:.3f}, Top-5 accuracy: - {top5_ave:.3f}, Bleu-4 score: - {bleu_score}\n'.format(loss_ave = loss_ave, top5_ave= top5_ave, bleu_score=bleu4_score))

    return bleu4_score


if __name__ == '__main__':
    main()