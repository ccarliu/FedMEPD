# FedMEPD

Implemention of paper "Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation"
![image](https://github.com/user-attachments/assets/7de88c25-8823-4c07-b774-d164a863fcc5)

## Dataset

BraTS 2018 and 2020: https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A?usp=sharing
HaNSeg: https://zenodo.org/records/7442914

## Train and test process
1. Download dataset
2. environment prepare
3. Training: Replace the datapath in the training command of run.sh, and run it. After training, the model will be stored in folder "./results". We provide pretrained models, data split as described in our paper, together with the code.
Testing: Replace the datapath and checkpoint path in the testing command of test.sh, and run it, which will do the test and store result in folder "test_results".
