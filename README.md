# Project Description
This project aims at exploring machine learning and deep learning solutions to binary and multi-class image classification tasks.

# Code Strcture
* [`A`](./A/): training and evaluation scripts of Task A
    * [`Model_A.py`](./A/Model_A.py): *ResNet-50* models and *CNNFS*
    * [`Train_A.py`](./A/Train_A.py): training script implemented with PyTorch
    * [`Test_A.py`](./A/Test_A.py): evaluation script
    * [`get_started_A.ipynb`](./A/get_started_A.ipynb): Training logs with Google Colab (softmax classifier)
    * [`classifier.py`](./A/classifier.py): SVM,KNN and DecisionTree classifier
    * [`Task_A.ipynb`](./A/Task_A.ipynb): Training logs with Google Colab (Machine learning classifier)

* [`B`](./B/): training and evaluation scripts of Task B
    * [`Model_B.py`](./B/Model_B.py): *ResNet-50* models with 3 methods
    * [`Train_B.py`](./B/Train_B.py): training script implemented with PyTorch
    * [`Test_B.py`](./B/Test_B.py): evaluation script
    * [`get_started_B.ipynb`](./B/get_started_B.ipynb): Training logs with Google Colab

* [`Datasets`](./Datasets/): data downloaded via https://medmnist.com/
    * [`pathmnist.nzp`](./Datasets/pathmnist.nzp): data for multi-class classification
    * [`pneumoniamnist.nzp`](./Datasets/pneumoniamnist.nzp): data for binary classification

* [`main.py`](./main.py): train the best model for each task

# Installation and Requirements
The code requires common Python environments for model training:
- Python 3.11.5
- PyTorch==1.3.1
- scikit-learn==1.3.2
- pandas==2.1.1
- numpy==1.26.1
- tqdm==4.66.1