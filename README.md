# EE655_Course_Project
Codes and Instructions for running EE655 Course Project

1. Download CamVid Dataset from this link: https://www.kaggle.com/datasets/carlolepelaars/camvid. Make sure that the files are in a folder called CamVid
2. Run the dataset.py script to make the following datasets:
   - Complex_CamVid_iHSV.
   - Complex_CamVid_inv_FFT. 
   - Complex_CamVid_RGB (Baseline).
   Note: You can tweak the transformations variable to get save other datasets as well(HSV real images for ex.)
4. Run complex_cam_vid.ipynb to train for complex inputs and evaluate jaccard score(Includes cells for both types of losses mentioned in write up)
5. Run cam_vid_baseline.ipynb to train for real inputs and evaluate jaccard score
6. To reproduce analysis results mentioned in the paper, run analysis.ipynb file. Make sure that you select the right model from the 'models/' folder, and your final output is correct


A few pre trained models are available here: https://drive.google.com/drive/folders/1AHWDIntsucMVuh8bfwzKxvdrb6Dnn_kF?usp=sharing

Link to write up(With references): https://drive.google.com/file/d/1pp4hzyUPcaKrHIyyByyU-OJJ8WqxnChF/view?usp=sharing

**Note**: Any error is most likely due to a non existent path, or loading model parameters into the wrong model(Ex.: Change C_ReLU function in case you want parameterized ReLU or not)
