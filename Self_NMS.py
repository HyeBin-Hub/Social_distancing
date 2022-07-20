import numpy as np

def soft_nms(dets, iou_thresh=0.3, sigma=0.5, thresh=0.001, method=2):

    N = dets.shape[0]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        temp_box = dets[i, :4]
        temp_score = dets[i, 4]
        temp_area = areas[i]
        pos = i + 1

        if i != N - 1:
            maxscore = np.max(dets[:, 4][pos:])
            maxpos = np.argmax(dets[:, 4][pos:])
        else:
            maxscore = dets[:, 4][-1]
            maxpos = -1

        if temp_score < maxscore:
            dets[i, :4] = dets[maxpos + i + 1, :4]
            dets[maxpos + i + 1, :4] = temp_box

            dets[i, 4] = dets[maxpos + i + 1, 4]
            dets[maxpos + i + 1, 4] = temp_score

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = temp_area

        xx1 = np.maximum(x1[i], x1[pos:])
        xx2 = np.minimum(x2[i], x2[pos:])
        yy1 = np.maximum(y1[i], y1[pos:])
        yy2 = np.minimum(y2[i], y2[pos:])

        w = np.maximum(xx2 - xx1 + 1, 0)
        h = np.maximum(yy2 - yy1 + 1, 0)

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[pos:] - overlaps)

        if method == 1:
            weight = np.ones(ious.shape)
            weight[ious > iou_thresh] = weight[ious > iou_thresh] - ious[ious > iou_thresh]
        elif method == 2:
            weight = np.exp(-ious * ious / sigma)
        else:
            weight = np.ones(ious.shape)
            weight[ious > iou_thresh] = 0

        dets[pos:, 4] = dets[pos:, 4] * weight

    idx = np.argwhere(dets[:, 4] > thresh)
    keep = idx.astype(int)[0]

    return keep

