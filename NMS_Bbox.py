import cv2
import numpy as np
import time

def get_NMS_Bbox(idxs, img, boxes,confidences):

    centroid = []
    NMS_boxes = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            box = boxes[i]
            # print(box)
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])

            right = int(left + width)
            bottom = int(top + height)

            NMS_boxes.append([left, top, right, bottom])

            centroid_x = ((left + right) / 2)
            centroid_y = ((top + bottom) / 2)
            centroid.append([int(centroid_x), int(centroid_y)])

    return centroid, NMS_boxes