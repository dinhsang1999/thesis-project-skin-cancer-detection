# SKIN DISEASE CLASSICICATION USING METADATA

| Some skin disease image |
| ---------- |
| <img src="./image/some.png" width="400">|

## 9 classes
|Melanoma (MEL)|
| ---------- |
|<img src="./image/MEL.png" width="200">|
|Actinic Keratosis (AK)|
| ---------- |
|<img src="./image/AK.png" width="200">|
Basel Cell Carcinoma (BCC)
| ---------- |
|<img src="./image/BCC.png" width="200">|
Benign Keratosis (BK)
| ---------- |
|<img src="./image/BKL.png" width="200">|
Melanocytic nevi (NV)
| ---------- |
|<img src="./image/NV.png" width="200">|
Vascular Skin Lesion (VASC)
| ---------- |
|<img src="./image/VASC.png" width="200">|
Squamous Cell Carcinoma (SCC)
| ---------- |
|<img src="./image/SCC.png" width="200">|
Dermatofibroma (DF)
| ---------- |
|<img src="./image/DF.png" width="200">|
Unknown (unknown)
| ---------- |
|<img src="./image/unknown.png" width="200">|

## Requirements
- Python >= 3.9

## Setup environment
Run this script to create a virtual environment and install dependency libraries
```bash
pip install -r requirements.txt
```

Set up: folder,tensorboard
```bash
bash init.bash
```

downloand data
```
link: https://www.kaggle.com/cdeotte/datasets 
note: full scale
```

***
## *<p style='color:cyan'>Edit training configuration in file src/ultils.py. You also can refer some set up from my_trial</p>*

## Train
```bash
python train.py
```
***
## Test
```bash
python test.py
```

## Predict
```bash
python predict.py
```
***
## Observe learning curve
```bash
tensorboard --logdir=log
```