import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2



class CustomDataset(Dataset):

    def __init__(self, data_root, data_name, split_name):

        self.split_name = split_name
        self.training_file = h5py.File(data_root + "/train36.hdf5", "r")
        self.validation_file = h5py.File(data_root + "/val36.hdf5", "r")
        self.training_features = self.training_file["image_features"]
        self.validation_features = self.validation_file["image_features"]

        # the distribution of image features used 
        with open(os.path.join(data_root, self.split_name + "_GENOME_DETS_" + data_name + ".json"), "r") as f:
            self.ifd = json.load(f)

        # the length of sequences
        with open(os.path.join(data_root, self.split_name + "_CAPLENS_" + data_name + ".json"), "r") as f:
            self.sequencelengths = json.load(f)

        # the sequences used
        with open(os.path.join(data_root, self.split_name + "_CAPTIONS_" + data_name + ".json"), "r") as f:
            self.sequences = json.load(f)


        # the size of dataset
        self.size = len(self.sequences)

        # sequences per image
        self.sequencesperimage = 5

    def __len__(self):
        return self.size

    def __getitem__(self, seqnum):

        #sequencelength and sequence for the item 
        sequencelength = torch.LongTensor([self.sequencelengths[seqnum]])
        sequence = torch.LongTensor(self.sequences[seqnum])

        # ifd object considering all distribution of images
        ifd = self.ifd[seqnum // self.sequencesperimage]
        #image features coming from bottum up attention
        if  ifd[0] == "v":
            imagefeatures = torch.FloatTensor(self.validation_features[ifd[1]])
            #salfeatures = torch.FloatTensor(sal)
        else:
            imagefeatures = torch.FloatTensor(self.training_features[ifd[1]])
            #salfeatures = torch.FloatTensor(sal)

        if self.split_name == "TRAIN":
            return imagefeatures, sequence, sequencelength
        else:
            # In order to calculate BLEU-4 score, we need all sequences generated per image
            sequences_generated = torch.LongTensor(self.sequences[(self.sequencesperimage*(seqnum//self.sequencesperimage)) : (self.sequencesperimage * (seqnum // self.sequencesperimage) + self.sequencesperimage)])
            return imagefeatures, sequence, sequencelength, sequences_generated

class SaliencyCustomDataset(Dataset):
    # returns saliency image features as well for the saliency map model.
    def __init__(self, data_root, data_name, split_name):

        self.split_name = split_name
        self.training_file = h5py.File(data_root + "/train36.hdf5", "r")
        self.training_features = self.training_file["image_features"]
        self.validation_file = h5py.File(data_root + "/val36.hdf5", "r")
        self.validation_features = self.validation_file["image_features"]

        # the distribution of image features used 
        with open(os.path.join(data_root, self.split_name + "_GENOME_DETS_" + data_name + ".json"), "r") as f:
            self.ifd = json.load(f)

        # the length of sequences
        with open(os.path.join(data_root, self.split_name + "_CAPLENS_" + data_name + ".json"), "r") as f:
            self.sequencelengths = json.load(f)
        # the sequences used
        with open(os.path.join(data_root, self.split_name + "_CAPTIONS_" + data_name + ".json"), "r") as f:
            self.sequences = json.load(f)


        # the size of dataset
        self.size = len(self.sequences)

        # sequences per image
        self.sequencesperimage = 5

        #
        self.list = os.listdir('./total_predictions')

    def __len__(self):
        return self.size

    def __getitem__(self, seqnum):

        #sequencelength and sequence for the item 
        sequencelength = torch.LongTensor([self.sequencelengths[seqnum]])
        sequence = torch.LongTensor(self.sequences[seqnum])

        # ifd object considering all distribution of images
        ifd = self.ifd[seqnum // self.sequencesperimage]
        sal = self.list[ifd[1]]
        sal_res = cv2.imread(os.path.join("total_predictions/"+sal), 0)

        sal_img = cv2.resize(sal_res, (64, 64), interpolation=cv2.INTER_NEAREST)

        salfeatures = torch.from_numpy(sal_img)
        salfeatures = salfeatures.reshape(-1)
        salfeatures = salfeatures.to(torch.float32)        
     
        #image features coming from bottum up attention
        if  ifd[0] == "v":
            imagefeatures = torch.FloatTensor(self.validation_features[ifd[1]])
        else:
            imagefeatures = torch.FloatTensor(self.training_features[ifd[1]])
        

        if self.split_name == "TRAIN":
            return imagefeatures, salfeatures, sequence, sequencelength
        else:
            # In order to calculate BLEU-4 score, we need all sequences generated per image
            sequences_generated = torch.LongTensor(self.sequences[(self.sequencesperimage*(seqnum//self.sequencesperimage)) : (self.sequencesperimage * (seqnum // self.sequencesperimage) + self.sequencesperimage)])
            return imagefeatures, salfeatures, sequence, sequencelength, sequences_generated


