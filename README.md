# RNN_HA_resnet50
source code: https://github.com/tzzcl/RNN-HA
Dataset: VeRi

the code is constructed at the direction of source code readme.

First train resnet50 as a pure classification and get the pretrained model.

Second train the RNN_HA model with the pretrained resnet model. 

  
Train: python train_rnnHa.py --training_data  /home/*/VeRi/image_train --txt_path ./train_label.txt --res_resume ./weight/checkpoint.pth.tar 

Evaluate(MAP,CMC): python evaluate.py --data_path /home/*/VeRi --resume ./models/model_epoch_40.pth

MAP: 63.1%    
CMC Scores    allshots      cuhk03  market1501
  top-1          32.1%       61.8%       89.7%
  top-5          42.7%       89.6%       95.0%
  top-10         48.8%       95.8%       97.0%
