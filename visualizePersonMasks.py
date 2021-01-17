import numpy as np
import cv2
import myLogger as log


def _draw_dotted_line(img, pt1, pt2, color):
    """
    Draws a dotted line between to points on an image
    :param img: The image to draw on
    :param pt1: Point 1 of the line (x, y)
    :param pt2: Point 2 of the line (x, y)
    :param color: The color of the line (same format as the image; BGR/RGB)
    """
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5  # Pythagorean Theorem
    pts = []
    number_of_points = int(dist / 7)

    # To get an even number of points
    if number_of_points % 2 == 1:
        number_of_points += 1

    for i in np.linspace(0, dist, number_of_points, endpoint=True):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        pts.append((x, y))

    for i in np.arange(0, len(pts) - 1, 2):
        # Only, if an endpoint for the dash if available
        if len(pts) > i + 1:
            cv2.rectangle(img, (pts[i][0] - 1, pts[i][1] - 1), pts[i + 1], color, -1)


def _draw_dashed_rect(img, pt1, pt2, color):
    """
    Draws a dashed rectangle
    :param img: The image to draw on
    :param pt1: Upper left point of the rectangle (x, y)
    :param pt2: Lower right point of the rectangle (x, y)
    :param color: The color of the line (same format as the image; BGR/RGB)
    """
    p1 = (pt1[0] - 1, pt1[1] - 1)
    p2 = (pt2[0] + 2, pt1[1] - 1)
    p3 = (pt2[0] + 2, pt2[1] + 2)
    p4 = (pt1[0] - 1, pt2[1] + 2)
    _draw_dotted_line(img, p1, p2, color)
    _draw_dotted_line(img, p2, p3, color)
    _draw_dotted_line(img, p1, p4, color)
    _draw_dotted_line(img, p4, p3, color)


def _draw_mask(image, mask, color, alpha=0.5):
    """
    Draws a binary mask onto an image
    :param image: The image to draw on
    :param mask: The binary mask to draw (Must be same size as the image)
    :param color: The color of the mask (same format as the image; BGR/RGB)
    :param alpha: Transparency of the mask
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def draw_masks_on_image(image, boxes, masks, group_ids_mask=None, group_ids_bounding_box=None,
                        draw_groups=False, draw_mask=False, draw_box=False, color=(0, 255, 255)):
    """
    Draws masks, hitboxes and their group ids on an image
    :param image: The image to draw on
    :param boxes: The bounding boxes of all masks
    :param masks: The binary masks
    :param group_ids_mask: The id of the group the mask have been assigned based on the mask content
    :param group_ids_bounding_box: The id of the group the mask have been assigned based on the space between
    mask and bounding box
    :param draw_groups: Whether to show the group ids
    :param draw_mask: Whether to draw the masks
    :param draw_box: Whether to draw the bounding boxes
    :param color: The color of the mask (same format as the image; BGR/RGB)
    """

    color_float = (color[0] / 255, color[1] / 255, color[2] / 255)

    # Number of instances
    nr_of_persons = boxes.shape[0]
    if not nr_of_persons:
        log.info("No persons on image")
    else:
        assert boxes.shape[0] == masks.shape[-1]

    masked_image = image.astype(np.uint8).copy()
    for i in range(nr_of_persons):
        if not np.any(boxes[i]):
            # Skip this instance. Has no bounding box. Likely lost in image cropping.
            continue

        # Draw hitbox
        if draw_box:
            y1, x1, y2, x2 = boxes[i]
            _draw_dashed_rect(masked_image, (x1, y1), (x2, y2), color)

        # Draw mask
        if draw_mask:
            mask = masks[:, :, i]
            masked_image = _draw_mask(masked_image, mask, color_float)

        # Print group number next to hitbox
        if draw_groups:
            text = '[' + str(i) + '] ' + str(group_ids_mask[i]) + '/' + str(group_ids_bounding_box[i])
            # Simulate outline
            cv2.putText(masked_image, text, (boxes[i][1], boxes[i][0] - 1), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(masked_image, text, (boxes[i][1], boxes[i][0] - 1), cv2.FONT_HERSHEY_PLAIN, 0.8, color,
                        1, cv2.LINE_AA)

    return masked_image
