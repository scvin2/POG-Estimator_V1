# POG-Estimator_V1
This repository contains a deep learning framework for estimating the Point-of-Gaze (POG) on a public display screen by utilizing a RGBD camera.
In this version of the project only Azure Kinect sensor is supported.
   
<p align="center">
  <img src="https://github.com/scvin2/POG-Estimator_V1/blob/master/Participant_gaze_example.jpg" width="500">
</p>

## Description
Currently many methods exist for estimating the point of POG on screen.
But, most of them are meant for use with personal computers and work only at close proximity.
They also don't work well when the user changes body position parallel to screen.
These issues make most of the previous methods not usable with display screen terminals present at public spaces like malls, airports, etc.

So, this project tries to predict POG for display screens present in public spaces while allowing the user to move parallely, closer or away from the screen. Additionaly the users can also move their head freely while using the system.

## Dataset
Gaze data of several participants has been collected for training this POG estimation system. 
During the data collection process, the particpants were instructed to make various eye and head movements while looking at marks displayed on a screen.
The participants where also instructed to change their position to a random location at random time.

The dataset will be made available online soon.

## Usage


## Requirements
+ Python >= 3.6
+ Keras == 2.2.4
+ Tensorflow == 1.13.1
+ OpenCV-Python
+ pyk4a
+ Platform: Linux

## Acknowledgments
