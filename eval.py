import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from scorer import compute_metrics
from datasets import *
from utils import *
from tqdm import tqdm


data_root = 'final_dataset'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = 'BEST_4BS20_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checpoint

mapping_file = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

mapping_file = os.path.join(data_root, 'WORDMAP_' + data_name + '.json')
with open(mapping_file, 'r') as j:
    mapping = json.load(j)
reverse_mapping = {value: key  for key, value in mapping.items()}
vocabSize = len(reverse_mapping)
# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location = device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval() # turn on evaluation mode 


def evaluate(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    groundtruths = list()  # true captions, list of lists as each image may have several truth captions
    predictions = list()  # predictions , list of predictions
    # DataLoader
    test_loader = torch.utils.data.DataLoader(CustomDataset(data_root, data_name, 'TEST'), batch_size=1, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())
    for i, (imagefeatures, sequence, sequencelength, sequences_generated) in enumerate(tqdm(test_loader)):      
        k = beam_size
        imagefeatures = imagefeatures.to(device)
        sequence = sequence.to(device)
        sequencelength = sequencelength.to(device)
        sequences_generated = sequences_generated.to(device)
        imagefeatures_mean = torch.mean(imagefeatures, dim=1).expand(k, 2048)  

        #Initialization step
        best_k_scores = torch.zeros(k, 1).to(device)
        prev_k_sequences  = torch.LongTensor([[mapping['<start>']]] * k).to(device)  
        k_sequences = prev_k_sequences
        hidden1, cell1 = decoder.init_hidden_state(k)  
        hidden2, cell2 = decoder.init_hidden_state(k) 

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        los = 1 #length of sequence
        
        # s is a number less than or equal to , because sequences are removed from this process once they hit <end>
        while True:
            embeddings = decoder.embedding(prev_k_sequences).squeeze(1)
            hidden1,cell1 = decoder.TD(torch.cat([hidden2,imagefeatures_mean,embeddings], dim=1),(hidden1,cell1)) 
            #attention_weighted_encoding = decoder.attModule(imagefeatures,hidden1) #(base model)
            attention_weighted_encoding, _ = decoder.attModule(imagefeatures,hidden1) 
            hidden2,cell2 = decoder.lang_layer(torch.cat([attention_weighted_encoding,hidden1], dim=1),(hidden2,cell2))
            scores = decoder.linear(hidden2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = torch.add(scores, best_k_scores.expand(scores.size()))  # (s, vocab_size)
            #scores = best_k_scores.expand_as(scores) + scores
            #Flatten the tensor, and select top k scores 
            if los == 1:
                best_k_scores, best_k_sequences = scores[0].topk(k, 0, True, True)  # (s)
            else:
                best_k_scores, best_k_sequences = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = torch.div(best_k_sequences, vocabSize, rounding_mode="trunc")  
            word_inds = best_k_sequences % vocabSize  # (s)

           #Update the sequences of words
            k_sequences = torch.cat([k_sequences[prev_word_inds], word_inds.unsqueeze(1)], dim=1) 

            # Which sequences are incomplete (didn't reach <end>)?
            continue_indices = [index for index, word in enumerate(word_inds) if word != mapping['<end>']]
            end_indices = list(set(range(len(word_inds))).difference(set(continue_indices)))

            if len(end_indices) > 0:
                complete_seqs.extend(k_sequences[end_indices].tolist())
                complete_seqs_scores.extend(best_k_scores[end_indices])
            
            k -= len(end_indices) 
            if  k == 0:
                break
            # shrink the sequences to consider only incomplete sequences
            hidden1 = hidden1[prev_word_inds[continue_indices]]
            cell1 = cell1[prev_word_inds[continue_indices]]
            hidden2 = hidden2[prev_word_inds[continue_indices]]
            cell2 = cell2[prev_word_inds[continue_indices]]
            imagefeatures_mean = imagefeatures_mean[prev_word_inds[continue_indices]]
            k_sequences = k_sequences[continue_indices]
            best_k_scores = best_k_scores[continue_indices].unsqueeze(1)
            prev_k_sequences = word_inds[continue_indices].unsqueeze(1)
            if los > 50:
                break
            los += 1
        
        x = complete_seqs_scores.index(max(complete_seqs_scores))
        best_sequence = complete_seqs[x]
    

        cap = sequences_generated[0].tolist()
        caps = list(
            map(lambda c: [reverse_mapping[w] for w in c if w not in {mapping['<start>'], mapping['<end>'], mapping['<pad>']}], cap))  # remove <start> and pads
        str_caps = [' '.join(c) for c in caps] #convert to string
        groundtruths.append(str_caps)
        prediction = ([reverse_mapping[w] for w in best_sequence if w not in {mapping['<start>'], mapping['<end>'], mapping['<pad>']}])
        prediction = ' '.join(prediction)
    
        predictions.append(prediction)


    # Calculate scores
    metrics_dict = compute_metrics(groundtruths, predictions)
    return metrics_dict





if __name__ == '__main__':
    beam_size = 5
    metrics_dict = evaluate(beam_size)
    print(metrics_dict)