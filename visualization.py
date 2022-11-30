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
        print(weight)
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

def visualize_att(image_path, best_sequence, best_attentions, bboxes, rev_word_map):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]


    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        attention = best_attentions[t, :]
        # alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        # if t == 0:
        #     plt.imshow(alpha, alpha=0)
        # else:
        #     plt.imshow(alpha, alpha=0.8)
        visualize_pred(im, boxes, att_weights)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

    

