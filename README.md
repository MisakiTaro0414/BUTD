# Image Captioning using Bottom-Up and Top-Down Attention
Implementation of Image Captioning model using Bottom-up attention and several modifications on the model. The original research paper, which is used as a reference for this project, is available at https://arxiv.org/abs/1707.07998

# Data preprocessing

In our implementation, we have used MSCOCO dataset. Please download MSCOCO dataset for training and validation (13GB and 6 GB) respectively. Please download Andrej Karpathy's training, validation, and test splits as well which contain the captions. Unzip those files and place the obtained files in 'data' folder in your workspace. 

We have used already obtained bottom-up features for training. Please download them using this link: https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip . Unzip the downloaded folder and move to the 'bottom_up_features' folder.

To obtain the best checkpoint files for different models, please download them using the following link: https://drive.google.com/file/d/1vVGdRQl7bUX4R97F-A3t4PFzv3PNsNq5/view?usp=share_link. Please unzip the file in your working directory to obtain ```results``` folder which contain best checkpoint files for several models implemented. 


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

Please note that checkpoint files require the models to have the path that were present when the checkpoint file was obtained. So, if you want to resume training please do the following file name and location changes: Move all models from '''models''' folder to root folder except '''saliency_model.py'''. Rename moved model files: ''' model.py -> model.py ''' , ''' arnet_sentinel_model.py -> ar_sen.py''', ''' simplified_model.py -> ablation_model.py ''', ''arnet_model.py -> ARNet_model.py''', ''' sentinel_model.py -> Adap_Att.py '''. As a result you must have the following structure:
```
|-- models
|   |-- saliency_model.py
|-- model.py
|-- ar_sen.py
|-- ablation_model.py
|-- ARNet_model.py 
|-- Adap_Att.py 
```



# Evaluation
For different models, we have different evaluation files. For experiments of base (replication) and ARNet model, please use ``` python eval.py ```. Please comment/uncomment the lines (line 67-68) in eval.py depending on using ARNet model or Base model. 
For models containing the Visual Sentinel, namely solely Sentinel and Sentinel+ARNet model, please use ``` python sentinel_eval.py ```. 
For simplified model, please use ``` python simplified_eval.py ```. 
For saliency model, please use ``` python saliency_eval.py ```. 

Please edit the variable checkpoint to the best checkpoint file path in python file to perform evaluation. Output will include BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, METEOR and ROUGE-L scores for the best checkpoint file.

# Inference
Due to CUDA compatibility issue, we did not manage to obtain end-to-end inference. To obtain the bottom-up features, run the inference.ipynb file in Google Colab. Please create a folder named "project" in Google Drive and add the content of "project" folder to that drive folder. After running '''inference.iypnb''' file it will save the bounding boxes as .pt in Google Drive. Load these boxes in test.py (line 114) and run command ``` python test.py ```. This will produce the best captioning of the image.











