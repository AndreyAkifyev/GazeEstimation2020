import cv2
import dlib
import numpy as np
import joblib
import sklearn

model_x = joblib.load('model_x.pkl') # model for predicting x-coordinate
model_y = joblib.load('model_y.pkl') # model for predicting y-coordinate

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2) # draw point on pupil
        return (cx, cy)
    except:
        pass

def nothing(x):
    pass

detector = dlib.get_frontal_face_detector() # face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # pretrained facial landmarks detector

left = [36, 37, 38, 39, 40, 41] # choosing only eye`s landmarks
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0) # initializing webcam
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) 
        thresh = cv2.dilate(thresh, None, iterations=4) 
        thresh = cv2.medianBlur(thresh, 3) 
        thresh = cv2.bitwise_not(thresh)
        center_right = contouring(thresh[:, 0:mid], mid, img)
        center_left = contouring(thresh[:, mid:], mid, img, True)
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
            
            cv2.line(img, center_right, tuple([int(lookpt_right_x * norm_right + shape[36][0] + 15),
                     int(lookpt_right_y * norm_right + shape[36][1])]), (0,255,0), 2)
            cv2.line(img, center_left, tuple([int(lookpt_left_x * norm_left + shape[42][0] + 13),
                     int(lookpt_left_y * norm_left + shape[42][1])]), (0,255,0), 2)
        except:
            pass
            
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to stop the program
        break
    
cv2.destroyAllWindows()
cap.release()
