These collection of scripts made up a pipline of modules where each modules can be run seperately. below will explain the use of each module and method to execute 

### 1_pre_processing_image

Perfrom resizing and padding of images to convert them into standard size of 128 x 128 and split into trainning & testing set based on given csv file 

* input : directory path for images,directory path to csvs files
* output : folders of images with "XXX_image_128","XXX_image_train" and "XXX_image_test" as folder names
* command :```python main.py -i /home/dj/NDSC/new_data/mobile2/mobile_image/ -o /home/dj/NDSC/new_data/mobile2/ -c mobile -d /home/dj/NDSC/csvs/```

### 1_pre_processing_text 

FillAttribute.py fills empty attribute columns by matching keywords on title column
* input : csv file 
* output : csv file
* command : ```python FillAttribute.py```

noDupTitleExt.py removes identical entries in the csv file 
* input : csv file 
* output : csv file
* command : python noDupTitleExt.py 
			
### 2_training_images (unfinished due to limit in computing power)
consist of 3 sub folders where each folder consist of 
* 1. train.py : main image training body 
* 2. design for each CNN 
* input : folder of images 
* command : ```python train.py -i /home/dj/NDSC/data/mobile/mobile_sample/ -o /home/dj/NDSC/data/mobile/ -c mobile -d /home/dj/NDSC/data/mobile/```

### 2_training_text 
perforom svm training for text analysis on category level and save the best models for each attribute into output directory
* inpupt : directory to csv files 
* output : a model for each attribut in for the format:  category_attribute.model 
* command : ```python main.py  -o /home/dj/NDSC/models/ -c beauty -d /home/dj/NDSC/csvs2/ -save true```

### 3_prediction_text 
performs prediction on category level using models saved during training 
* input: trained model 
* output : a csv file
* command : ```python main.py -i /home/dj/NDSC/models/fashion/ -c fashion -d /home/dj/NDSC/csvs2/kaggle/```

### 4_submision
convert predicted cvs into submission format 
* input : predicted csv for all 3 category 
* output : a csv file for submission 
* command : ```python main.py -d /home/dj/NDSC/test_csv/ -o /home/dj/NDSC/submission/data_info_val_sample_submission.csv -c all```
