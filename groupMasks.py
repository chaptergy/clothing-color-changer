import numpy as np
import colorFinder
import myLogger as log


def hue_array_to_string_array(hue_arr):
    """
    Converts an array of OpenCV hues (0..180; as well as -1 and -2) to the name of colors
    """
    hue_names = ["red   ", "orange", "yellow", "lime  ", "green ", "teal  ", "cyan  ", "lblue ", "dblue ", "purple",
                 "magent", "pink  "]
    hue_array_360 = hue_arr.copy()
    # Convert OpenCV's hues (0..180) to standard (0..360) hues, whole ignoring white and black (-1 and -2)
    hue_array_360[hue_arr > 0] = hue_array_360[hue_arr > 0] * 2

    string_array = hue_array_360.copy().astype(np.str)
    for i in range(30, 331, 30):
        string_array[((hue_array_360 >= i - 15) & (hue_array_360 < i + 15))] = hue_names[int(i / 30)]

    string_array[((hue_array_360 >= 345) | ((hue_array_360 < 15) & (hue_array_360 >= 0)))] = \
        hue_names[0]  # Red at the top (>=345) and at the bottom (< 15)
    string_array[hue_array_360 == -1] = "white "
    string_array[hue_array_360 == -2] = "black "

    hue_array_360 = hue_array_360.astype(np.str)
    if len(hue_array_360.shape) == 1:
        hue_array_360 = np.array(['%4s' % t for t in hue_array_360], dtype=np.str)
    elif len(hue_array_360.shape) == 2:
        hue_array_360 = np.array([['%4s' % t for t in el] for el in hue_array_360], dtype=np.str)
    hue_array_360[hue_arr < 0] = ''

    string_array = np.core.defchararray.add(string_array, hue_array_360)

    return string_array


def group_masks_by_dyn_color(image, mask_rcnn_result, inside=True, nr_of_color_groups=4, grouping_precision=3):
    """
    Groups the masks by similar colors, with the colors being dynamic and depending on the most used color
    :param image: The image in BGR format
    :param mask_rcnn_result: The dictionary returned by Mask RCNN containing masks and rois
    :param inside: Whether the inside of the mask should be used (True) or the space between mask an bounding box
    :param nr_of_color_groups: Into how many groups the colors in each mask should be grouped
    :param grouping_precision: The precision with which the masks will be put into groups (the larger the precision,
    the more similar the colors have to be to be grouped into the same group
    """
    all_dominant_colors = np.array(
        colorFinder.find_dominant_hue_colors_for_each_mask(image, mask_rcnn_result, inside, nr_of_color_groups))

    colors_to_consider = 4  # Only consider the most prominent colors
    max_mismatches = 1  # How many colors are allowed to not match for it to be still in the same group

    group_info = np.empty((0, 4, min(colors_to_consider, all_dominant_colors.shape[1])), dtype=int)
    groups = np.empty(all_dominant_colors.shape[0], dtype=int)

    hue_threshold = int(180 / nr_of_color_groups / grouping_precision)

    for mIndex in range(0, all_dominant_colors.shape[0]):

        dominant_colors = all_dominant_colors[mIndex, 0:min(colors_to_consider, all_dominant_colors.shape[1])]
        log.debug("Colors", ("inside" if inside else "around"), "Mask " + str(mIndex) + ":",
                  hue_array_to_string_array(dominant_colors[:, 0]))

        # There is no group yet
        if group_info.shape[0] == 0:
            group_info = np.append(group_info, [[
                dominant_colors[:, 0].copy(),  # Median values of the individual hues
                dominant_colors[:, 0].copy(),  # Sum of the individual hues (For median calculation)
                np.ones(len(dominant_colors[:, 0]), dtype=int),  # Amount of individual hues (For median calculation)
                dominant_colors[:, 1].copy()  # Nr of pixels in this hue
            ]], axis=0)
            groups[mIndex] = group_info.shape[0] - 1

        else:
            probable_group = None
            for i, existingGroup in enumerate(group_info):

                best_existing_matches = np.full((len(dominant_colors[:, 0]), 2), (-1, -1))
                for j in range(0, len(dominant_colors[:, 0])):

                    for k in range(0, len(existingGroup[0])):
                        if dominant_colors[j, 0] < 0:  # for black and white
                            if dominant_colors[j, 0] == existingGroup[0, k]:
                                best_existing_matches[j] = (k, 0)
                                break
                        else:
                            if existingGroup[0, k] < 0:
                                continue

                            if is_similar_hue(dominant_colors[j, 0], existingGroup[0, k], hue_threshold):
                                hue_diff = abs(dominant_colors[j, 0] - existingGroup[0, k])

                                # If no match found yet or hue difference is lower than previous match
                                if (best_existing_matches[j, 0] == -1) | (hue_diff < best_existing_matches[j, 1]):
                                    best_existing_matches[j] = (k, hue_diff)

                missing_matches = len(best_existing_matches[best_existing_matches[..., 0] == -1])

                # Max n colors not matching and most prominent color is one of the most prominent in group
                if (missing_matches <= max_mismatches) & (
                        best_existing_matches[0, 0] <= int(colors_to_consider / 2) - 1) & \
                        (best_existing_matches[0, 0] != -1):
                    if probable_group is None:
                        # (Group-ID, nr of matched colors, array of matched colors)
                        probable_group = (i, len(best_existing_matches) - missing_matches, best_existing_matches[:, 0])
                    elif missing_matches < probable_group[1]:
                        probable_group = (i, len(best_existing_matches) - missing_matches, best_existing_matches[:, 0])

            if probable_group is None:  # Create new group
                group_info = np.append(group_info, [[
                    dominant_colors[:, 0].copy(),  # Median values of the individual hues
                    dominant_colors[:, 0].copy(),  # Sum of the individual hues (For median calculation)
                    np.ones(len(dominant_colors[:, 0]), dtype=int),  # Amount of individual hues (For median calculation)
                    dominant_colors[:, 1].copy()  # Nr of pixels in this hue
                ]], axis=0)
                groups[mIndex] = group_info.shape[0] - 1

            else:
                groups[mIndex] = probable_group[0]

                # Calculate new medians
                for colIndex, match in enumerate(probable_group[2]):
                    if dominant_colors[colIndex, 0] > 0:
                        group_info[probable_group[0], 1][match] += dominant_colors[
                            colIndex, 0]  # Sum of the hues
                    group_info[probable_group[0], 2][match] += 1  # Sum of the individual hues
                    group_info[probable_group[0], 0][match] = int(
                        group_info[probable_group[0], 1][match] / group_info[probable_group[0], 2][match])  # Median val
                    group_info[probable_group[0], 3][match] += dominant_colors[colIndex, 1]  # Nr of pixels

                group_info[probable_group[0], :] = np.transpose(group_info[probable_group[0], :,
                                                                np.flip(group_info[probable_group[0], 3].argsort(),
                                                                        axis=0)])  # Sort by frequency

    return groups, group_info[:, 0]


def is_similar_hue(hue1, hue2, threshold):
    """
    Checks if two hues are similar
    :param hue1: Hue to check (0..360)
    :param hue2: Hue to check (0..360)
    :param threshold: Threshold until what delta hues are considered similar
    :return: Whether the hues are similar
    """
    if abs(hue1 - hue2) < threshold:
        return True
    elif abs(hue1 - 360) + hue2 < threshold:
        return True
    elif abs(hue2 - 360) + hue1 < threshold:
        return True
    else:
        return False
