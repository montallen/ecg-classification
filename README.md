# Dilated Convolutional Networks for Automatic Features Extraction from ECG

This repo contains the code and a poster of my final master project at the Technical University of Denmark, Lyngby. The thesis work is done in collaboration between DTU and the Laboratory of Experimental Cardiology, BMI, KU.

The poster in the repo was presented at the 44th Annual Conference of the International Society of Computerized Electrocardiology (ISCE2019).

Note that this is still an ongoing project. 


### Authors
- Alessandro Montemurro
  DTU Compute, Biomedicinsk Institute, Unicersity of Copenhagen.
 
 
### Description
Manual features engineering can represent a limitation in machine learning algorithms because we are not sure we can find the most appropriate
set of featues to use for a specific classification. An end-to-end approach is proposed, where the raw ECG data is given as input to the network.
Classification is done according to 4 different targets: Sinus Bradychardia, Fast Heart-rate, Left Ventriculat Hypertrophy, Gender.

Class Activation Map (CAM) is used to open the black box and understand where the network was looking at to carry out the classification.
The folder "cam_plots" contains some plots of ECG. The red regions are the regions that the network judged most important.

![cam](https://github.com/AllenMont/ecg-classification/blob/master/cam_plots/21.png)

More details about CAM can be found in http://cnnlocalization.csail.mit.edu/
