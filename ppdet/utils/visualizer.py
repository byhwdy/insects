import numpy as np
from PIL import Image, ImageDraw

from .colormap import colormap

__all__ = ['visualize_results']


def visualize_results(image,
                      im_id,
                      catid2name,
                      threshold=0.5,
                      bbox_results=None):
    """
    Visualize bbox and mask results
    """
    if bbox_results:
        image = draw_bbox(image, im_id, catid2name, bbox_results, threshold)
    return image

def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill=color)

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image
