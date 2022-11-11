import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import  DecoderAttModule
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True 

# Data parameters
data_root = 'final_dataset'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
embSize = 1024  # dimension of word embeddings
decoderSize = 1024  # dimension of decoder RNN
attentionSize = 1024  # dimension of attention linear layers
dropout = 0.5
 

# Training parameters
continue_epoch = 0
epochs = 50  # number of epochs to train for (if early stopping is not triggered)
bad_epochs = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 100
workers = 1  # for data-loading; right now, only 1 works with h5py
best_bleu4 = 0.  # BLEU-4 score right now
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, bad_epochs, checkpoint, continue_epoch, data_name, mapping

    # Read word map
    mapping_file = os.path.join(data_root, 'WORDMAP_' + data_name + '.json')
    with open(mapping_file, 'r') as j:
        mapping = json.load(j)


    if checkpoint is None:
        decoder = DecoderAttModule(attentionSize, embSize,decoderSize, len(mapping), dropout)

        optimizer = torch.optim.Adamax(params=filter(lambda x: x.requires_grad, decoder.parameters()))

    else:
        checkpoint = torch.load(checkpoint)
        continue_epoch = checkpoint['epoch'] + 1
        bad_epochs = checkpoint['bad_epochs']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        optimizer = checkpoint['optimizer']
       
    # Move to GPU, if available
    decoder = decoder.to(device)

    #Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(CustomDataset(data_root, data_name, 'TRAIN'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(CustomDataset(data_root, data_name, 'VAL'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
 
    # Epochs
    for epoch in range(continue_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if bad_epochs == 20:
            break
        if bad_epochs > 0 and bad_epochs % 8 == 0:
            for parameters in optimizer.param_groups:
                parameters['lr'] = parameters['lr'] * 0.75
            print("Learning rate decreased by 25% percent")
    
        # One epoch's training
        train(train_loader, decoder, criterion, optimizer, epoch)

        # One epoch's validation
        bleu4_score = validate(val_loader, decoder, criterion)
     
        if bleu4_score > best_bleu4_score:
            best_bleu4_score = bleu4_score
            bad_epochs = 0
            
        else:
            bad_epochs += 1
            print("\nEpochs since last improvement: %d\n" % (bad_epochs,))

        # Save checkpoint
        save_checkpoint(data_name, epoch, bad_epochs, decoder, optimizer, bleu4_score)


def train(train_loader, decoder, criterion, optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # Activate Dropout and BatchNormalization

    batch_time = 0  # forward prop. + back prop. time
    data_time = 0  # data loading time
    losses = 0 # loss (per word decoded)
    loss_count = 0
    top5accs = 0  # top5 accuracy
    top5counts = 0

    time_start = time.time()

    # Batches
    for i, (imagefeatures, sequence, sequencelength) in enumerate(train_loader):
        data_time.update(time.time() - time_start)

        # Move to GPU, if available
        imagefeatures = imagefeatures.to(device)
        sequence = sequence.to(device)
        sequencelength = sequencelength.to(device)

        # Forward prop.
        preds, sorted_sequence, decode_lengths, sort_indexes = decoder(imagefeatures, sequence, sequencelength)
  
        #Max-pooling across predicted words across time steps for discriminative supervision

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        labels = sorted_sequence[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        preds, _ = pack_padded_sequence(preds, decode_lengths, batch_first=True)
        labels, _ = pack_padded_sequence(labels, decode_lengths, batch_first=True)

        # Calculate loss

        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
	
        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        top5 = topk_accuracy(preds, labels, 5)
        top5accs += top5
        top5counts += sum(decode_lengths)
        top5_ave = top5accs / top5counts
        losses += loss
        loss_count += sum(decode_lengths)
        loss_ave = losses/loss_count
        batch_time = time.time() - time_start
        time_start = time.time()

        # Print status
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Data Load Time {data_time:.3f})\t'
                  'Loss {loss:.4f} ({loss_ave:.4f})\t'
                  'Top-5 Accuracy {top5:.3f} ({top5_ave:.4f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=loss, loss_ave =loss_ave, 
                                                                          top5 = top5, top5_ave = top5_ave))


def validate(val_loader, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = 0
    losses = 0
    top5accs = 0

    time_start = time.time()

    groundtruths = []  # references (true captions) for calculating BLEU-4 score
    predictions = []  # hypotheses (predictions)

    # Batches
    with torch.no_grad(): 
        for i, (imagefeatures, sequence, sequencelength, sequences_generated) in enumerate(val_loader):

            # Move to device, if available
            imagefeatures = imagefeatures.to(device)
            sequence = sequence.to(device)
            sequencelength = sequencelength.to(device)

            preds, sorted_sequence, decode_lengths, sort_indexes = decoder(imagefeatures, sequence, sequencelength)


            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            labels =sorted_sequence[:, 1:]
            

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            copy_preds = preds.clone()
            preds, _ = pack_padded_sequence(preds, decode_lengths, batch_first=True)
            labels, _ = pack_padded_sequence(labels, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(preds, labels)

            # Keep track of metrics
            top5 = topk_accuracy(preds, labels, 5)
            top5accs += top5
            top5counts += sum(decode_lengths)
            top5_ave = top5accs / top5counts
            losses += loss
            loss_count += sum(decode_lengths)
            loss_ave = losses/loss_count
            batch_time = time.time() - time_start

            start = time.time()

            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time:.3f})\t'
                      'Loss {loss:.4f} ({loss_ave:.4f}) )\t'
                      'Top-5 Accuracy {top5:.3f} ({top5_ave:.3f})\t'.format(i, len(val_loader), batch_time,
                                                                                loss, loss_ave, top5, top5_ave))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            sequences_generated = sequences_generated[sort_indexes]  # because images were sorted in the decoder
            for i in range(sequences_generated.shape[0]):
                cap = sequences_generated[i].tolist()
                caps = list(map(lambda x: [word for word in x if word not in {mapping['<start>'], mapping['<pad>']}], cap))
                groundtruths.append(caps)


            _, best_preds = torch.max(copy_preds, dim=2)
            best_preds = best_preds.tolist()
            temp = []
            for i, prediction in enumerate(best_preds):
                temp.append(best_preds[i][:decode_lengths[i]])  # remove pads
            best_preds = temp
            predictions.extend(best_preds)


    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(predictions, groundtruths)
    bleu4 = round(bleu4, 4)

    print('\n * LOSS - {loss_avg:.3f}, TOP-5 ACCURACY - {top5_avg:.3f}, BLEU-4 - {bleu}\n'.format(loss_ave, top5_ave, bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()