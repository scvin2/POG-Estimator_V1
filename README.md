# POG-Estimator_V1
This repository contains a deep learning framework for estimating the Point-of-Gaze (POG) on a public display screen by utilizing a RGBD camera.
In this version of the project only Azure Kinect sensor is supported.


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
1. Clone this repository

2. Download the dataset into the data directory

3. Extract the dataset into the extracted_data directory by running the following command
```
python3 extract_gaze_data.py
```

4. Train the POG estimator model using train_gaze_pointer.py file.
```
python3 train_gaze_pointer.py
```

5. The weights file will be saved into the gaze_pointer_checkpoints directory.

6. After training, the model can be tested using the tester file.
```
python3 pointer_interface_test.py
```

## Requirements
+ Python >= 3.6
+ Keras == 2.2.4
+ Tensorflow == 1.13.1
+ OpenCV-Python
+ pyk4a
+ Platform: Linux

## Acknowledgments
The face detector model was from [RetinaFace](https://github.com/deepinsight/insightface), the facial landmark detector was from [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) and the eye landmark detector model was based on [ELG](https://github.com/swook/GazeML).
