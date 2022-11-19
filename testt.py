from image import *
import cv2

IMG_FILE = "./COCO_train2014_000000150367.jpg"

cfg_file = "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool_wrapper.yaml"

im = cv2.imread(IMG_FILE)
model = FasterRCNNBottomUp(cfg_file)

boxes = model(im)

print(boxes)
print(boxes.shape)