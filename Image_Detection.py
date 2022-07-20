import cv2
import numpy as np
import time
from Social_distancing.objectInfo import get_object_info
from Social_distancing.NMS_Bbox import get_NMS_Bbox
from Social_distancing.draw_run import draw
from Social_distancing.NMS import py_cpu_nms
from Social_distancing.Self_NMS import soft_nms

img_file= "//Social_distancing/data/pedestrians_1_Moment.jpg"
img=cv2.imread(img_file)

pretrain_model="C:/Users/hyebin/PycharmProjects/55/Social_distancing/pretrained/yolov3.weights"
config_file= "//Social_distancing/pretrained/yolov3.cfg"

cv_net=cv2.dnn.readNetFromDarknet(config_file,pretrain_model)

img_H, img_W = img.shape[:2]

layer_name=cv_net.getLayerNames()
output_layer_ind=cv_net.getUnconnectedOutLayers()
output_layer_name=[layer_name[layer[0]-1] for layer in output_layer_ind]

blob=cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
cv_net.setInput(blob)

start_time=time.time()
results=cv_net.forward(output_layer_name)

boxes, class_ids, confidences = get_object_info(results, img_W, img_H,0.01)

# idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

# NMS
dets=np.array([i+[j] for i,j in zip(boxes,confidences)])
idxs_nms=py_cpu_nms(dets, 0.7)

# Self_NMS
dets=np.array([i+[j] for i,j in zip(boxes,confidences)])
idxs_soft_nms = soft_nms(dets, iou_thresh=0.7, sigma=0.5, thresh=0.0001, method=2)

img_nms=img.copy()
img_soft_nms=img.copy()

centroid_nms, NMS_boxes_nms = get_NMS_Bbox(idxs_nms, img_nms, boxes,confidences)
centroid_soft_nms, NMS_boxes_soft_nms = get_NMS_Bbox(idxs_soft_nms, img_soft_nms, boxes,confidences)

img_nms=draw(img_nms,centroid_nms,NMS_boxes_nms)
img_soft_nms=draw(img_soft_nms,centroid_soft_nms,NMS_boxes_soft_nms)

cv2.imshow("img",img_nms)
cv2.imshow("img_soft_nms",img_soft_nms)

cv2.imwrite("//Social_distancing/result/pedestrians_1_Moment.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

