import cv2
import numpy as np
import time

def get_object_info(results, img_w, img_h,conf_threshold):
    boxes = []
    class_ids = []
    confidences = []

    for feature_map in results:
        for feature_map_info in feature_map:
            score = feature_map_info[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if confidence > conf_threshold and class_id == 0:
                box = feature_map_info[:4]

                c_x_nrom = box[0]
                c_y_nrom = box[1]
                w_norm = box[2]
                h_norm = box[3]

                c_x = int(c_x_nrom * img_w)
                c_y = int(c_y_nrom * img_h)
                w = int(w_norm * img_w)
                h = int(h_norm * img_h)
                xmin = int(c_x - w / 2)
                ymin = int(c_y - h / 2)

                boxes.append([xmin, ymin, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    return boxes, class_ids, confidences