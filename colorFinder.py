import numpy as np
import cv2
import operator
from math import ceil

bwThreshold = 80


def find_dominant_hue_colors(img, img_rcnn, inside=True, nr_of_color_groups=4):
    """
    Find the most dominant hues either within the masks or within the bounding boxes of the masks
    :param img: The source image
    :type img: BGR numpy array
    :param img_rcnn: RCNN data of the image
    :param inside: Whether to use only the inside of the masks (True) or the bounding box of each mask (False)
    :param nr_of_color_groups: Into how many clusters the colors (excluding black and white) should be grouped
    :return: The grouped colors sorted from most used to least used
    """
    dominant_colors, white, black = make_hue_histogram(img, img_rcnn, inside)

    dominant_colors = group_hues(dominant_colors, nr_of_color_groups)

    dominant_colors.append((-1, white))  # white
    dominant_colors.append((-2, black))  # black

    dominant_colors = sorted(dominant_colors, key=operator.itemgetter(1), reverse=True)

    return dominant_colors


def make_hue_histogram(img, img_rcnn, inside):
    """
    Counts the number of pixels for all hues and black and white
    :param img: The source image
    :type img: BGR numpy array
    :param img_rcnn: RCNN data of the image
    :param inside: Whether to use only the inside of the masks (True) or the bounding box of each mask (False)
    :return: hue histogram, # of white pixels, # of black pixels
    """
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # hues is an array of tuples (hue value, nr of pixels)
    hues = []

    counter = 0
    while counter != 179:
        hues.append((counter, 0))
        counter += 1
    white = 0
    black = 0

    for j in range(0, img_rcnn['masks'].shape[0]):
        for k in range(0, img_rcnn['masks'].shape[1]):

            if inside:
                # check if the pixel is inside mask
                if 1 in img_rcnn['masks'][j, k]:
                    # If saturation > 51 it is probably a color
                    if hsv_image[j, k][1] > 51:
                        hues[hsv_image[j, k][0]] = (hues[hsv_image[j, k][0]][0], hues[hsv_image[j, k][0]][1] + 1)
                    # Otherwise it is probably black or white
                    else:
                        if hsv_image[j, k][2] > 128:
                            white += 1
                        else:
                            black += 1
            else:
                for i, box in enumerate(img_rcnn['rois']):
                    if j > box[0] & j < box[2] & k > box[1] & k < box[3]:
                        if 1 not in img_rcnn['masks'][j, k]:
                            # If saturation > 51 it is probably a color
                            if hsv_image[j, k][1] > 51:
                                hues[hsv_image[j, k][0]] = (
                                    hues[hsv_image[j, k][0]][0], hues[hsv_image[j, k][0]][1] + 1)
                            # Otherwise it is probably black or white
                            else:
                                if hsv_image[j, k][2] > 128:
                                    white += 1
                                else:
                                    black += 1

    return hues, white, black


def group_hues(dominant_colors, nr_of_color_groups):
    """
    Groups a hue histogram into a specified number of groups
    :param dominant_colors: The hue histogram data
    :param nr_of_color_groups: How many groups to create
    :return: Grouped dominant colors
    """
    block_size = 180 / nr_of_color_groups
    max_value = sorted(dominant_colors, key=operator.itemgetter(1), reverse=True)
    start_value = max_value[0][0] - (block_size / 2) % 180

    dominant_color_blocks = []
    counter = max_value[0][0]

    # Create empty color blocks
    for x in range(0, nr_of_color_groups):
        dominant_color_blocks.append((counter % 180, 0))
        counter += block_size

    # Group existing data into the new empty blocks
    for index in range(0, 180):
        dominant_color_blocks[int((index - (index % block_size)) / block_size)] = (
            int(dominant_color_blocks[int((index - (index % block_size)) / block_size)][0]),
            dominant_color_blocks[int((index - (index % block_size)) / block_size)][1] +
            dominant_colors[int((start_value + index) % 180)][1])

    return dominant_color_blocks


def find_dominant_hue_colors_for_each_mask(img, img_rcnn, inside=True, nr_of_color_groups=4):
    """
    Find the dominant color in each mask independently
    :param img: Source image
    :param img_rcnn: RCNN data
    :param inside: Whether to use only the inside of the masks (True) or the bounding box of each mask (False)
    :param nr_of_color_groups: Into how many clusters the colors (excluding black and white) should be grouped
    :return: The grouped colors for each mask sorted from most used to least used
    """
    dominant_colors = make_hue_histogram_for_each_mask(img, img_rcnn, inside)

    dominant_colors = group_hues_for_each_mask(dominant_colors, nr_of_color_groups)
    dominant_colors.view('i4,i4').sort(order=['f1'], axis=1)  # sort colors by frequency (descending)

    return np.flip(dominant_colors, axis=1)


def make_hue_histogram_for_each_mask(img, img_rcnn, inside):
    """
    Counts the number of pixels for all hues and black and white for every mask independently
    :param img: The source image
    :type img: BGR numpy array
    :param img_rcnn: RCNN data of the image
    :param inside: Whether to use only the inside of the masks (True) or the bounding box of each mask (False)
    :return: hue histogram for each mask
    """
    global bwThreshold
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # hues is an array of tuples (hue value, nr of pixels)
    hues = np.arange(0, img_rcnn['rois'].shape[0] * 182, 0.5).reshape((img_rcnn['rois'].shape[0], 182, 2))
    hues = hues % 182
    hues[hues % 1 != 0] = 0
    hues = hues.astype(int)

    masks = img_rcnn['masks']  # Shorten variable name
    if not inside:
        masks_flattened = np.zeros(masks.shape[:-1], dtype=int)
        masks_flattened[
            np.any(masks, axis=-1)] = 1  # Binary mask which merged all masks (so pixels in masks won't be counted)

    rois = img_rcnn['rois']
    box = np.zeros_like(masks, dtype=int)

    for i in range(0, masks.shape[-1]):
        if inside:
            count_values = np.bincount(
                img[(img[..., 1] > bwThreshold) & (masks[..., i] == 1), 0])  # count each hue value
            hues[i, ..., 1] = np.pad(count_values, (0, 182 - len(count_values)),
                                     mode='constant')  # pad with zeros, so there are 362 values in the array
            hues[i, 180] = (
                -1, len(img[(img[..., 1] <= bwThreshold) & (img[..., 2] > 128) & (masks[..., i] == 1)]))  # white
            hues[i, 181] = (
                -2,
                len(img[(img[..., 1] <= bwThreshold) & (img[..., 2] <= 128) & (masks[..., i] == 1)]))  # black
        else:
            box[rois[i, 0]:rois[i, 2], rois[i, 1]:rois[i, 3], i] = 1  # bounding box to binary mask

            count_values = np.bincount(img[(img[..., 1] > bwThreshold) & (box[..., i] == 1) &
                                           (masks_flattened[...] == 0), 0])  # count each hue value
            hues[i, ..., 1] = np.pad(count_values, (0, 182 - len(count_values)),
                                     mode='constant')  # pad with zeros, so there are 362 values in the array
            hues[i, 180] = (-1, len(img[(img[..., 1] <= bwThreshold) & (img[..., 2] > 128) &
                                        (box[..., i] == 1) & (masks_flattened[...] == 0)]))  # white
            hues[i, 181] = (-2, len(img[(img[..., 1] <= bwThreshold) & (img[..., 2] <= 128) &
                                        (box[..., i] == 1) & (masks_flattened[...] == 0)]))  # black

    return hues


def group_hues_for_each_mask(dominant_colors, nr_of_color_groups):
    """
    Groups a hue histogram into a specified number of groups for each mask independently
    :param dominant_colors: The hue histogram data by mask
    :param nr_of_color_groups: How many groups to create
    :return: Grouped dominant colors for each mask
    """
    block_size = ceil(180 / nr_of_color_groups)
    dominant_color_blocks = np.zeros((dominant_colors.shape[0], nr_of_color_groups + 2, 2), dtype=np.int32)
    max_value = np.max(dominant_colors[..., :180, 1])
    start_value = max_value - (block_size / 2) % 180

    for mIndex in range(0, dominant_colors.shape[0]):
        counter = max_value
        for x in range(0, nr_of_color_groups):
            dominant_color_blocks[mIndex, x] = (counter % 180, 0)
            counter += block_size

        for index in range(0, 180):
            block_index = int((index - (index % block_size)) / block_size)
            dominant_color_blocks[mIndex, block_index, 1] += dominant_colors[
                mIndex, int((start_value + index) % 180), 1]

        dominant_color_blocks[mIndex, -2:] = dominant_colors[mIndex, -2:]  # Copy last two values (white and black)

    return dominant_color_blocks
