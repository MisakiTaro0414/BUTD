import os
import io
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import pickle
import numpy as np
import utils
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'width_image', 'height_image', 'num_boxes', 'boxes', 'features']
input_file = 'trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
file_train = 'train36.hdf5'
file_validation = 'val36.hdf5'
file_tr_ind = 'train36_imgid2idx.pkl'
file_val_ind = 'val36_imgid2idx.pkl'
training_file = 'train_ids.pkl'
validation_file = 'val_ids.pkl'

num_features = 2048
box_limit = 36


if __name__ == '__main__':
    
    howto_train = h5py.File(file_train, "w")
    
    howto_validate = h5py.File(file_validation, "w")

    if os.path.exists(training_file) and os.path.exists(validation_file):
        with open(training_file, "rb") as f:
            images_training = pickle.load(f)
        with open(validation_file, "rb") as f:
            images_validation = pickle.load(f)

        #images_training = pickle.load(open(training_file, "rb"))
        #images_validation = pickle.load(open(validation_file, "rb"))
    else:
        images_training = utils.load_imageid('../data/train2014')
        images_validation = utils.load_imageid('../data/val2014')
        pickle.dump(images_training, open(training_file, 'wb'),protocol=2)
        pickle.dump(images_validation, open(validation_file, 'wb'),protocol=2)

    ind_train = {}
    ind_val = {}

    feats_trainimg = howto_train.create_dataset(
        'image_features', (len(images_training), box_limit, num_features), 'f')

    bb_trainimg = howto_train.create_dataset(
        'image_bb', (len(images_training), box_limit, 4), 'f')

    spatialfeats_trainimg = howto_train.create_dataset(
        'spatial_features', (len(images_training), box_limit, 6), 'f')

    bb_valimg = howto_validate.create_dataset(
        'image_bb', (len(images_validation), box_limit, 4), 'f')

    feats_valimg = howto_validate.create_dataset(
        'image_features', (len(images_validation), box_limit, num_features), 'f')

    spatialfeats_valimg = howto_validate.create_dataset(
        'spatial_features', (len(images_validation), box_limit, 6), 'f')

    count_train = 0
    count_val = 0

    print("loading TSV file...")
    with open(input_file, "r") as file_tsv:
        
        reader = csv.DictReader(file_tsv, delimiter='\t', fieldnames=FIELDNAMES)
        
        for objecter in tqdm(reader):

            objecter['num_boxes'] = int(objecter['num_boxes'])

            id_image = int(objecter['image_id'])
            width_image = float(objecter['width_image'])
            height_image = float(objecter['height_image'])

            # Bounding Box generation
            boundingboxes = np.frombuffer(
                base64.decodebytes(bytes(objecter['boxes'], encoding="utf8")),
                dtype=np.float32).reshape((objecter['num_boxes'], -1))

            w_box = boundingboxes[:, 2] - boundingboxes[:, 0]
            h_box = boundingboxes[:, 3] - boundingboxes[:, 1]

            # Scaling image dimensions
            s_w = w_box / width_image
            s_h = h_box / height_image
            
            # Scaling bounding box dimensions
            s_x = boundingboxes[:, 0] / width_image
            s_y = boundingboxes[:, 1] / height_image

            """"""
            w_box = w_box[..., np.newaxis]
            h_box = h_box[..., np.newaxis]

            s_w = s_w[..., np.newaxis]
            s_h = s_h[..., np.newaxis]
            
            s_x = s_x[..., np.newaxis]
            s_y = s_y[..., np.newaxis]
            """"""
            
            feats_space = np.concatenate(
                (s_x,
                 s_y,
                 s_x + s_w,
                 s_y + s_h,
                 s_w,
                 s_h),
                axis=1)

            if id_image in images_training:
                images_training.remove(id_image)
                ind_train[id_image] = count_train
                
                bb_trainimg[count_train, :, :] = boundingboxes
                
                feats_trainimg[count_train, :, :] = np.frombuffer(
                    base64.decodebytes(bytes(objecter['features'], encoding="utf8")),
                    dtype=np.float32).reshape((objecter['num_boxes'], -1))
                
                spatialfeats_trainimg[count_train, :, :] = feats_space
                count_train += 1
            
            elif id_image in images_validation:
                images_validation.remove(id_image)
                ind_val[id_image] = count_val
                
                bb_valimg[count_val, :, :] = boundingboxes
                
                feats_valimg[count_val, :, :] = np.frombuffer(
                    base64.decodebytes(bytes(objecter['features'], encoding="utf8")),
                    dtype=np.float32).reshape((objecter['num_boxes'], -1))
                
                spatialfeats_valimg[count_val, :, :] = feats_space
                count_val += 1
                
            else:
                assert False, 'Unknown image id: %d' % id_image

    if len(images_training) != 0:
        print('Warning: training_image_id is not empty')

    if len(images_validation) != 0:
        print('Warning: validation_image_id is not empty')

    pickle.dump(ind_train, open(file_tr_ind, 'wb'))
    pickle.dump(ind_val, open(file_val_ind, 'wb'))
    howto_train.close()
    howto_validate.close()
    print("done!")
