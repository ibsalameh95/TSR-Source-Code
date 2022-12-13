# TSR-Source-Code
Reproducing the results of the paper: Artificial intelligence quantified tumour-stroma ratio is an independent predictor for overall survival in resectable colorectal cancer
Zhao, K., Li, Z., Yao, S., Wang, Y., Wu, X., Xu, Z., ... & Liu, Z. (2020). Artificial intelligence quantified tumour-stroma ratio is an independent predictor for overall survival in resectable colorectal cancer. EBioMedicine, 61, 103054.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

> please download the data from this link https://zenodo.org/record/4024676#.Y49odS8RpQI, move downloaded folders into project_path/data/ directory. 

## Project files
 
* __train_cnn.py__
	* Takes train and validation datasets, model parameters and start model training and validation. 
	* Data visualization for the training and validation results.
* __test_cnn.py__ 
	* Takes testing dataset, test the model. 
	*  Data visualization for the testing results.

## Training

```train
python train_cnn.py --train_dataset_dir=data/TSR-CRC-Training-set-part1 --vall_dataset_dir=data/TSR-CRC-Training-set-part3 --num_epochs=10 --batch_size=64 --learning_rate=3e-4
```


```test
python test_cnn.py --test_dataset_dir=data/TSR-CRC-Test-set2 --batch_size=64
```
