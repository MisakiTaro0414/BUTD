# Image Captioning using Bottom-Up and Top-Down Attention
Implementation of Image Captioning model using Bottom-up attention and several modifications on the model. The original research paper, which is used as a reference for this project, is available at https://arxiv.org/abs/1707.07998

# Data preprocessing

In our implementation, we have used MSCOCO dataset. Please download MSCOCO dataset for training and validation (13GB and 6 GB) respectively. Please download Andrej Karpathy's training, validation, and test splits as well which contain the captions. Unzip those files and place the obtained files in 'data' folder in your workspace. 

We have used already obtained bottom-up features for training. Please download them using this link: https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip . Unzip the downloaded folder and move to the 'bottom_up_features' folder.

Next type the command: 
```bash
python bottom-up_features/tsv.py
```
Please create a folder named 'final_dataset' and move obtained files to there. 

Lastly, type the command:
```bash
python create_input_files.py
```
It will create the useful files for training/validation/testing. Please move those files to 'final_dataset' folder.

# Training 
Type this command for training from scratch:
```bash
python train.py
```
If you want to resume your training, please edit the variable checkpoint to the checkpoint file path in train.py. We have experimented several models in our project. If you want to experiment these models, please go through train.py and comment/uncomment the lines to specify the model to be used. 


# Evaluation









