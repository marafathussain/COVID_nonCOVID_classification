# COVID vs. nonCOVID Classification Task using DenseNet121
Running this code needs [MONAI](https://github.com/Project-MONAI/MONAI) framework installed. I used the preliminary dataset from [COVID19-Challenge.eu](https://www.covid19challenge.eu/covid-19-toy-datasets/?utm_medium=email&_hsmi=89256177&_hsenc=p2ANqtz-9qW4BFNLuv9Uh1QUHg4AOKONAUEpxZ4PFQjkQ3frWiKJFOA-8lBa-ReX1I6HRCjqjMv2gOvf68nkmRO2UILFc44BFD5w&utm_content=89256177&utm_source=hs_email). 72 CT volumes were used for training and 24 CT volumes were used for validation. 

## Data preprocessing
All the CT volumes were resampled to a common voxel dimension of 1.6mm X 1.6mm X 3.2mm. The labels (1: COVID, 0: nonCOVID) were generated from the existance of opacity masks in the segmentation data. The lung area was also masked using the provided segmentation mask. 

## Accuracy
The model classifies beween COVID and nonCOVID cases with ~96% accuracy.

## Pre-trained model
The pretrained model can be downloaded from this [link](https://drive.google.com/file/d/11HzTu79q05Pr9IJZQN9fIcZk4ErapweA/view?usp=sharing).
