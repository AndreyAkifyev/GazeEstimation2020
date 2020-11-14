# GazeEstimation2020

This is my first computer vision project, hope you like it! =)

In the folder "models" you can find PupilNet and pretrained weights for it for pupil detection. Also there are pretrained SVR models for predicting gaze direction. Folder "notebooks" contains Jupyter notebooks with my attempts to train models. As training dataset I used [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/tutorial.html).

In the example.avi you can see how the eye tracker works.

To install necessary libraries you may run:
```
pip install -r requirements.txt
```

To download the pretrained model for detecting the facial landmarks run:
```
bash get_pretrained_model.sh
```

To run the demo:
```
python EyeTracking.py
```
