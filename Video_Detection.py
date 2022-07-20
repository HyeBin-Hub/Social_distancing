import cv2
import numpy as np
import time
from Social_distancing.objectInfo import get_object_info
from Social_distancing.NMS_Bbox import get_NMS_Bbox
from Social_distancing.draw_run import draw
from Social_distancing.Self_NMS import soft_nms

def get_video_detect(video_path, video_output_path,cv_net,conf_thrshold,num_threshold,standard_distance=100):

  cap=cv2.VideoCapture(video_path)

  codec=cv2.VideoWriter_fourcc(*'XVID')

  video_size=(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

  video_fps=cap.get(cv2.CAP_PROP_FPS)

  video_write=cv2.VideoWriter(video_output_path,codec,video_fps,video_size)

  frame_cnt=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print("총 frame 갯수 : ",frame_cnt)

  while True:

    hasFrame,ImgFrame=cap.read()

    if not hasFrame:
      print("x")
      break

    ImgFrame_H, ImgFrame_W = ImgFrame.shape[:2]

    layer_name=cv_net.getLayerNames()
    output_layer_ind=cv_net.getUnconnectedOutLayers()
    output_layer_name=[layer_name[layer[0]-1] for layer in output_layer_ind]

    blob=cv2.dnn.blobFromImage(ImgFrame,1/255.0,(416,416),swapRB=True,crop=False)
    cv_net.setInput(blob)

    start_time=time.time()
    results=cv_net.forward(output_layer_name)

    boxes, class_ids, confidences = get_object_info(results,ImgFrame_W,ImgFrame_H,0.5)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thrshold, num_threshold)

    # Self_NMS
    # dets = np.array([i + [j] for i, j in zip(boxes, confidences)])
    # idxs_soft_nms = soft_nms(dets, iou_thresh=0.7, sigma=0.5, thresh=0.0001, method=2)

    # centroid_soft_nms, NMS_boxes_soft_nms = get_NMS_Bbox(idxs_soft_nms, ImgFrame, boxes, 0.5)

    centroid, NMS_boxes = get_NMS_Bbox(idxs,ImgFrame, boxes,0.5)

    img = draw(ImgFrame, centroid, NMS_boxes,standard_distance)

    print('Detection 수행 시간:',round(time.time()-start_time,2),"초")
    video_write.write(img)

  cap.release()
  video_write.release()



pretrain_model="C:/Users/hyebin/PycharmProjects/55/Social_distancing/pretrained/yolov3.weights"
config_file= "//Social_distancing/pretrained/yolov3.cfg"

cv_net=cv2.dnn.readNetFromDarknet(config_file,pretrain_model)

### 1
video_1_path= "//Social_distancing/data/pedestrians_1.mp4"
video_1_output_path= "//Social_distancing/result/pedestrians_output.mp4"

# get_video_detect(video_1_path,video_1_output_path,cv_net, 0.5, 0.5,100)

### 2
video_2_path= "//Social_distancing/data/pedestrian_2.mp4"
video_2_output_path= "//Social_distancing/result/pedestrians_output2.mp4"
get_video_detect(video_2_path,video_2_output_path,cv_net, 0.5, 0.5,100)


