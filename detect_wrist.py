import cv2 as cv
import numpy as np
import argparse


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()

    return frame, points

def cropHands(points):
    cap = cv.VideoCapture(args.input if args.input else 0)

    # Initialize frame counter
    cnt = 0

    # Some characteristics from the original video
    w_frame, h_frame = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv.CAP_PROP_FPS), cap.get(cv.CAP_PROP_FRAME_COUNT)

    # define croping values
    minusX = 50
    minusY = 100
    h = 150
    w = 150

    # output
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('result.avi', fourcc, fps, (w, h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        x = points[cnt][0] - minusX
        y = points[cnt][1] - minusY
        
        cnt += 1 # Counting frames

        # Avoid problems when video finish
        if ret==True:
            # Croping the frame
            crop_frame = frame[y:y+h, x:x+w]

            # Percentage
            xx = cnt *100/frames
            print(int(xx),'%')

            # I see the answer now. Here you save all the video
            out.write(crop_frame)
        else:
            break

    cap.release()
    out.release()
    
def getWristPoints():
    points = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            out.release()
            break
        output, tmp_points = poseDetector(frame)
        points.append(tmp_points[7])
        out.write(output)
    
    out.release()
    print("Done processing video and points are ready")
    return points

def fixPoints(points):
    
    if(points[0] == None): # to do: improve it
        points[0] = (0, 0)

    for i in range(len(points)):
        if points[i] == None:
            points[i] = points[i-1]

    new_points = []
    new_points.extend(points)
    new_points.extend(points)

    return new_points

def argsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = argsParser()
    inWidth = args.width
    inHeight = args.height
    thr = args.thr
    
    cap = cv.VideoCapture(args.input if args.input else 0)
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    print("Processing Video...")

    brokenWristPoints = getWristPoints()
    fixedPoints = fixPoints(brokenWristPoints)
    cropHands(fixedPoints)