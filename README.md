# FedMEPD

Implemention of paper "Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation"
![image](https://github.com/user-attachments/assets/7de88c25-8823-4c07-b774-d164a863fcc5)

## Dataset

1. [BraTS 2018 and 2020](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A?usp=sharing)
2. [HaNSeg](https://zenodo.org/records/7442914)

## Train and test process
1. Download dataset
2. Environment prepare: pip install -r requirements.txt
3. Training: Replace the datapath in the training command of run.sh, and run it. The training code will utilize four GPUs to simultaneously train the server and client, and the training time is approximately 30 hours. After training, the model will be stored in folder "./results".
4. Testing: Replace the datapath and checkpoint path in the testing command of test.sh, and run it, which will do the test and store results in folder "test_results". Our pretrainied model is available [here](https://drive.google.com/drive/folders/1lAW-QM_zA_fw_7Zd1iBZowr0SKaqLSJz?usp=sharing).
