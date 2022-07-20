import cv2
from Social_distancing.objectInfo import get_object_info
from Social_distancing.NMS_Bbox import get_NMS_Bbox
from Social_distancing.draw_run import draw
import numpy as np

def py_cpu_nms(dets, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    keep = []

    conf_sort = scores.argsort()[::-1]

    while conf_sort.size > 0:
        i = conf_sort[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[conf_sort[1:]])
        y11 = np.maximum(y1[i], y1[conf_sort[1:]])
        x22 = np.minimum(x2[i], x2[conf_sort[1:]])
        y22 = np.minimum(y2[i], y2[conf_sort[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[conf_sort[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        conf_sort = conf_sort[idx + 1]

    return keep


