import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle


def create_input_files(dataset,karpathy_json_path,captions_per_image, min_word_freq,output_folder,max_len=100):

    assert dataset in {'coco'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    
    with open(os.path.join(output_folder,'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)
        
    with open(os.path.join(output_folder,'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)
    
    # Read image paths and captions for each image
    train_image_captions = []
    val_image_captions = []
    test_image_captions = []
    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()
    
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue
        
        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])

        if img['split'] in {'train', 'restval'}:
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t",train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v",val_data[image_id]))
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            if image_id in val_data:
                val_image_det.append(("v",val_data[image_id]))
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            if image_id in val_data:
                test_image_det.append(("v",val_data[image_id]))
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_det) == len(train_image_captions)
    assert len(val_image_det) == len(val_image_captions)
    assert len(test_image_det) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
   
    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
        
    
    for impaths, imcaps, split in [(train_image_det, train_image_captions, 'TRAIN'),
                                   (val_image_det, val_image_captions, 'VAL'),
                                   (test_image_det, test_image_captions, 'TEST')]:
        enc_captions = []
        caplens = []
        
        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
                
            # Sanity check
            assert len(captions) == captions_per_image
            
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)
        
        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
    
    # Save bottom up features indexing to JSON files
    with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_image_det, j)
        
    with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_image_det, j)
        
    with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_image_det, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def save_checkpoint(data_name, epoch, bad_epochs, decoder, optimizer, bleu4, best):
    state = {'epoch': epoch,
             'bad_epochs': bad_epochs,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if best:
        torch.save(state, 'BEST_' + str(epoch) + filename)


def save_checkpoint_sentinel(data_name, epoch, bad_epochs, decoder, optimizer, bleu4, best):
    state = {'epoch': epoch,
             'bad_epochs': bad_epochs,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': optimizer}
    filename = 'Sentinel_checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if best:
        torch.save(state, 'BEST_' + str(epoch) + filename)


def save_checkpoint_arnet(data_name, epoch, bad_epochs, decoder, optimizer, bleu4, best):
    
    state = {'epoch': epoch,
             'bad_epochs': bad_epochs,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': optimizer}
    filename = 'ARNet_checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if best:
        torch.save(state, 'BEST_' + str(epoch) + filename)

def save_checkpoint_arnet_sentinel(data_name, epoch, bad_epochs, decoder, optimizer, bleu4, best):
    
    state = {'epoch': epoch,
             'bad_epochs': bad_epochs,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': optimizer}
    filename = 'ARNet+Sentinel_checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if best:
        torch.save(state, 'BEST_' + str(epoch) + filename)

def save_checkpoint_saliency(data_name, epoch, bad_epochs, decoder, optimizer, bleu4, best):
    
    state = {'epoch': epoch,
             'bad_epochs': bad_epochs,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': optimizer}
    filename = 'Saliency_checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if best:
        torch.save(state, 'BEST_' + str(epoch) + filename)



def save_checkpoint_simplified(data_name, epoch, bad_epochs, decoder, optimizer, bleu4, best):
    
    state = {'epoch': epoch,
             'bad_epochs': bad_epochs,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': optimizer}
    filename = 'Simplified_checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if best:
        torch.save(state, 'BEST_' + str(epoch) + filename)



