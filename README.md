# GazeEstimation2020

This is my first computer vision project.
Files model_x.pkl and model_y.pkl are pretrained SVR model for predicting the gaze vector by eye landmarks.
Notebook sandbox.ipynb contains my attempts to extract features from dataset UnityEyes and to train the SVR model with best hyperparameters.

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
