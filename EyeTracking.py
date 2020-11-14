import cv2
import dlib
import os
import numpy as np
import joblib
import sklearn
import torch
from eye_sample import EyeSample
from eye_prediction import EyePrediction
from models.PupilNet import PupilNet_v2
import torch.nn as nn

model_x = joblib.load('models/model_x.pkl') # model for predicting x-coordinate of gaze vector
model_y = joblib.load('models/model_y.pkl') # model for predicting y-coordinate of gaze vector
model = PupilNet_v2()                # model for predicting pupil center
model.load_state_dict(torch.load('models/pupilnet_v5.pt'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def predict_pupil(eyes, ow=160, oh=96):
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img/255.0], dtype=torch.float32).to(device)
            pupil = model(x.view(1, 1, 96, 160))
            pupil = np.asarray(pupil.cpu().numpy())
            assert pupil.shape == (1, 2)
            tmp = pupil[0][0]
            pupil[0][0] = pupil[0][1] / 2
            pupil[0][1] = tmp / 2
            pupil = pupil * np.array([oh/48, ow/80])
            temp = np.zeros((1, 3))
            if eye.is_left:
                temp[:, 0] = ow - pupil[:, 1]
            else:
                temp[:, 0] = pupil[:, 1]
            temp[:, 1] = pupil[:, 0]
            temp[:, 2] = 1.0
            pupil = temp
            assert pupil.shape == (1, 3)
            pupil = np.asarray(np.matmul(pupil, eye.transform_inv.T))[:, :2]
            assert pupil.shape == (1, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=pupil, gaze=None))
    
    return result

def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(42, 45, True), (36, 39, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        if is_left:
            eye_image = np.fliplr(eye_image)
            
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes



detector = dlib.get_frontal_face_detector() # face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # pretrained facial landmarks detector

left = [36, 37, 38, 39, 40, 41] # choosing only eye`s landmarks
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0) # initializing webcam
ret, img = cap.read()
shape = None

while True:
    ret, img = cap.read()
    orig_frame = img.copy()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        
        eye_samples = segment_eyes(gray, shape)
        pupil_predicts = predict_pupil(eye_samples)
        
        left_eyes = list(filter(lambda x: x.eye_sample.is_left, pupil_predicts))
        right_eyes = list(filter(lambda x: not x.eye_sample.is_left, pupil_predicts))

        center_left = int(round(left_eyes[0].landmarks[0][0])), int(round(left_eyes[0].landmarks[0][1]))
        center_right = int(round(right_eyes[0].landmarks[0][0])), int(round(right_eyes[0].landmarks[0][1]))
        
        cv2.circle(img, center_left, 2, (0, 0, 255), -1)
        cv2.circle(img, center_right, 2, (0, 0, 255), -1)
        
        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        
        norm_right = np.sqrt(np.sum((np.array([shape[36][0], shape[36][1]]) - \
                                         np.array([shape[39][0], shape[39][1]])) ** 2))
        norm_left = np.sqrt(np.sum((np.array([shape[42][0], shape[42][1]]) - \
                                         np.array([shape[45][0], shape[45][1]])) ** 2))
        try:
            ldmks_right = (np.array([[shape[36][0], shape[36][1]],
                          [shape[37][0], shape[37][1]],
                          [shape[38][0], shape[38][1]],
                          [shape[39][0], shape[39][1]],
                          [shape[40][0], shape[40][1]],
                          [shape[41][0], shape[41][1]],
                          list(center_right)]) - [shape[36][0], shape[36][1]]) / norm_right
            ldmks_left = (np.array([[shape[42][0], shape[42][1]],
                          [shape[43][0], shape[43][1]],
                          [shape[44][0], shape[44][1]],
                          [shape[45][0], shape[45][1]],
                          [shape[46][0], shape[46][1]],
                          [shape[47][0], shape[47][1]],
                          list(center_left)]) - [shape[42][0], shape[42][1]]) / norm_left
            
            lookpt_right_x = model_x.predict(ldmks_right.reshape(1, -1))
            temp = np.append(ldmks_right.reshape(1, -1), lookpt_right_x)
            lookpt_right_y = model_y.predict(temp.reshape(1, -1))
            
            lookpt_left_x = model_x.predict(ldmks_left.reshape(1, -1))
            temp = np.append(ldmks_left.reshape(1, -1), lookpt_left_x)
            lookpt_left_y = model_y.predict(temp.reshape(1, -1))
            
            cv2.line(img, center_right, tuple([int(lookpt_right_x * norm_right * 1.5 + shape[36][0]+3),
                     int(lookpt_right_y * norm_right + shape[36][1])]), (0,255,0), 2)
            cv2.line(img, center_left, tuple([int(lookpt_left_x * norm_left * 1.5 + shape[42][0]+3),
                     int(lookpt_left_y * norm_left + shape[42][1])]), (0,255,0), 2)
        except:
            pass
            
    cv2.imshow('eyes', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to stop the program
        break
    
cv2.destroyAllWindows()
cap.release()
