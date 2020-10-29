# GazeEstimation2020

This is my first computer vision project, hope you like it!

Files model_x.pkl and model_y.pkl are pretrained SVR models for predicting the gaze vector using eye landmarks.
Notebook sandbox.ipynb contains my attempts to extract features from dataset [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/tutorial.html) and to train the SVR model with best hyperparameters.

In the example.mp4 you can see how the eye tracker works.

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
