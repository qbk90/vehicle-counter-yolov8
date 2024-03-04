import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model = YOLO("yolov8s.pt")

# Run the model on the GPU instead of CPU
model.to("cuda")

#MOUSE EVENT FUNCTION
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         colorsBGR = [x, y]
#         print(colorsBGR)


cv2.namedWindow("RGB")
# cv2.setMouseCallback("RGB", RGB)

cap = cv2.VideoCapture("highway.mp4")

my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

# list of detected objects to be tracked
class_of_interest = ["car", "bus", "truck"]

# area variable is the list of points which form
# the polygon area of interest expressed (x_i,y_y)
area = [(270, 238), (294, 280), (592, 226), (552, 207)]

# create tracker instance
tracker = Tracker()

vehicles_passed = set()

count = 0
while True:

    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    # saves the data of the detected objects
    # x1, y1, x2, y2, confidence, class
    a = results[0].boxes.data

    # print(results[0].boxes)

    # create an empty list to later store the coordinates
    # of the objects to be tracked
    tracked_objects_list = []

    # save the info abt the coordinates in a pandas data frame
    # the result object needs to be transfered from the GPU memory to the CPU
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    # print(px)

    for index, row in px.iterrows():
        # print(row)
        x1, y1 = int(row[0]), int(row[1])
        x2, y2 = int(row[2]), int(row[3])
        class_id = int(row[5])
        class_name = str(class_list[class_id])
        if class_name in class_of_interest:
            tracked_objects_list.append([x1, y1, x2, y2])

    tracked_objects = tracker.update(tracked_objects_list)

    for bbox in tracked_objects:
        # take the coordinates of the bounding box and id of each object
        # that's being tracked
        x3, y3, x4, y4, id = bbox

        # create a center point of the object
        center_x = int(x3 + x4) // 2
        center_y = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
        cv2.putText(
            frame,  # location of text
            str(id),  # text str
            (x3, y3),  # coordinates of text
            cv2.FONT_HERSHEY_COMPLEX,  # text font
            0.5,  # scale
            (255, 0, 0),  # color
            1,  # thickness
        )
        # display the center point on the frame
        cv2.circle(
            frame,  # location of circle
            (center_x, center_y),  # coordinates of circle
            4,  # radius
            (0, 0, 255),  # color
            -1,  # thickness. -1 means fill (?)
        )
        vehicles_passed.add(id)

    # cv2.polylines(
    #     frame,  # location of polygon
    #     [
    #         np.array(area, np.int32)
    #     ],  # point coordinates of polygon turned into a numpy array
    #     True,  # whether the polygon is closed
    #     (255, 255, 0),  # color of polygon
    #     2,  # thinkness
    # )

    counter = len(vehicles_passed)

    # cv2.putText(
    #     frame,  # location of text
    #     f"Total vehicles passed: {counter}",  # text str
    #     (300, 300),  # coordinates of text
    #     cv2.FONT_HERSHEY_PLAIN,  # text font
    #     1.5,  # scale
    #     (255, 255, 255),  # color
    #     1,  # thickness
    # )

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
