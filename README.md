# This is the code for running the Mask_RCNN model for DSB_2018. 
## Data prepration. 
1. Download the test and train data:
    1. https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/archive/master/stage1_train.zip
    2. https://raw.githubusercontent.com/AakashSudhakar/2018-data-science-bowl/master/compressed_files/stage1_test.zip 
    3. copy then to /public/dsb_2018_data/stage1_train and /public/dsb_2018_data/stage1_test respectively. 
2. Git clone neptune code from here: https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/archive/master/stage1_train.zip
    1. Run neptune run main.py -- prepare_metadata
    2. Run neptune run main.py -- prepare_masks
5. Unzip our zip file. it will have a folder dsb_2018. 
6. cd to the dsb_2018 and git clone Mask_RCNN from here. https://github.com/matterport/Mask_RCNN.git
## Run training 
    1. 	cd to dsb_2018 and do git checkout result-0.461.
	2.  then run train_mask_rcnn.py script
## Run testing
    1. cd to dsb_2018 and run ./predict.py. 
    2. the result will be saved in /output/ folder. the name of the file is submission.csv
