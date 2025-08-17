import cv2

def draw_boxes(img, dets, color=(0,255,0)):
    im = img.copy()
    for d in dets:
        x1,y1,x2,y2 = map(int, d["bbox"])
        cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
        label = f'{d["class_name"]}:{d["conf"]:.2f}'
        cv2.putText(im, label, (x1, max(y1-6,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return im
