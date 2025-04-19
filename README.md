# EE655_Course_Project
Codes and Instructions for running EE655 Course Project

1. Download CamVid Dataset from this link: https://www.kaggle.com/datasets/carlolepelaars/camvid
2. Run the dataset.py script to make the following datasets:
     a. Complex_CamVid_iHSV
     b. Complex_CamVid_inv_FFT
     c. Complex_CamVid_RGB (Baseline)
   Note: You can tweak the transformations variable to get save other datasets as well(HSV real images for ex.)
3. Run complex_cam_vid.ipynb to train for complex inputs and evaluate jaccard score(Includes cells for both types of losses mentioned in write up)
4. Run cam_vid_baseline.ipynb to train for real inputs and evaluate jaccard score
5. To reproduce analysis results mentioned in the paper, run analysis.ipynb file. Make sure that you select the right model from the 'models/' folder, and your final output is correct
