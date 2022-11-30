import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CustomDataset(Dataset):
    """
    The custom datset class to be used for DataLoader.
    """

    def __init__(self, data_root, data_name, split_name, transform=None):
        """
        data_root: the root of folder that contains datasets
        data_name: base name of processed datasets
        split_name: 'TRAIN', 'VAL', or 'TEST' which shows 
        transform: transformations used, default of None
        """
        self.split_name = split_name
        self.validation_file = h5py.File(data_root + "/val36.hdf5", "r")
        self.validation_features = self.validation_file["image_features"]
        self.training_file = h5py.File(data_root + "/train36.hdf5", "r")
        self.training_features = self.training_file["image_features"]

        # the distribution of image features used 
        with open(os.path.join(data_root, self.split_name + "_GENOME_DETS_" + data_name + ".json"), "r") as f:
            self.ifd = json.load(f)

        # the length of sequences
        with open(os.path.join(data_root, self.split_name + "_CAPLENS_" + data_name + ".json"), "r") as f:
            self.sequencelengths = json.load(f)
        # the sequences used
        with open(os.path.join(data_root, self.split_name + "_CAPTIONS_" + data_name + ".json"), "r") as f:
            self.sequences = json.load(f)

        self.transform = transform

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
        else:
            imagefeatures = torch.FloatTensor(self.training_features[ifd[1]])
        

        if self.split_name == "TRAIN":
            return imagefeatures, sequence, sequencelength
        else:
            # In order to calculate BLEU-4 score, we need all sequences generated per image
            sequences_generated = torch.LongTensor(self.sequences[(self.sequencesperimage*(seqnum//self.sequencesperimage)) : (self.sequencesperimage * (seqnum // self.sequencesperimage) + self.sequencesperimage)])
            return imagefeatures, sequence, sequencelength, sequences_generated


