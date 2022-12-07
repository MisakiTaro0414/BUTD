import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2

from PIL import Image

cmap = cm.get_cmap('jet')
cmap.set_bad(color="k", alpha=0.0)

def attention_bbox_interpolation(im, bboxes, att):
    softmax = att
    assert len(softmax) == len(bboxes)

    img_h, img_w = im.shape[:2]
    opacity = np.zeros((img_h, img_w), np.float32)
    for bbox, weight in zip(bboxes, softmax):
        x1, y1, x2, y2 = bbox
        opacity[int(y1):int(y2), int(x1):int(x2)] += weight.item()
    opacity = np.minimum(opacity, 1)

    opacity = opacity[..., np.newaxis]   
     
    vis_im = np.array(Image.fromarray(cmap(opacity, bytes=True), 'RGBA'))
    vis_im = vis_im.astype(im.dtype)
    vis_im = cv2.addWeighted(im, 0.7, vis_im, 0.5, 0)
    vis_im = vis_im.astype(im.dtype)
    
    return vis_im

def visualize_pred(im, boxes, att_weights):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    b,g,r,a = cv2.split(im)           # get b, g, r
    im = cv2.merge([r,g,b,a])

    M = min(len(boxes), len(att_weights))
    im_ocr_att = attention_bbox_interpolation(im, boxes[:M], att_weights[:M])
    plt.imshow(im_ocr_att)

def visualize_att(image_path, best_sequence, best_attentions, bboxes, reverse_mapping):

    image = Image.open(image_path)
    image = image.resize([14 * 24,  14* 24], Image.LANCZOS)
    words = [reverse_mapping[ind] for ind in best_sequence]
    for t in range(len(words)):
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=15)
        attention = best_attentions[t]
        visualize_pred(im, bboxes, attention)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
        plt.show()

    

