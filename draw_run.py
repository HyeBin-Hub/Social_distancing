import cv2
import numpy as np

def draw(img,centroid,boxess,standard_distance):
    tpyes = "safe"
    color_b = (255, 153, 51)
    color_t = (255, 128, 0)

    for i in range(len(centroid)):
        # print(boxess[i][0],boxess[i][1],boxess[i][2],boxess[i][3])
        tpyes = "safe"
        color_b = (255, 153, 51)
        color_t = (255, 128, 0)
        for j in range(len(centroid)):

            if i != j:
                distance = np.sqrt((centroid[i][0] - centroid[j][0]) ** 2 + (centroid[i][1] - centroid[j][1]) ** 2)
                if distance < standard_distance:
                    # print("i",i,"j",j,centroid[i],centroid[j],round(distance),[boxess[i][0],boxess[i][1],boxess[i][2],boxess[i][3]])
                    tpyes = "unsafe"
                    color_b = (51, 51, 255)
                    color_t = (102, 102, 255)
        # print(boxess[i][0],boxess[i][1],boxess[i][2],boxess[i][3])
        cv2.rectangle(img, (boxess[i][0], boxess[i][1]), (boxess[i][2], boxess[i][3]), color_b, thickness=3)
        cv2.putText(img, tpyes, (boxess[i][0], boxess[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_t, 2)

    return img