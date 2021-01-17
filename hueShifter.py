import cv2
import numpy as np


def shift_hsv(bgr_image, hsv_values, threshold, index):
    """
    Shifts a hue on an image
    :param bgr_image: The image in BGR format
    :param hsv_values: List of lists (current hsv ([0..360],[0..255],[0..255]), target hue ([0..360],[0..255],[0..255]))
                 The target hue tuple can be replaced by one of the following string values:
                     'Regenbogen,n': Fade the colors in a rainbow, so for every single frame the hue will increment by n
                     'Strobe,n,m', Flash the colors, so for every n frames the hue will be incremented by m
    :type hsv_values: List of lists containing ((int, int, int),(int, int, int)/string))
    :param threshold: How many hue values above and below the current hue, saturation and value will be shifted.
                      The best results are when hue is low, saturation medium and value high (e.g. [30,60,160])
    :type threshold: (int, int, int)
    :param index: Index of the current frame. Is used for Regenbogen (rainbow) and Strobe effect
    :return: The shifted image
    """

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_arr = np.array(hsv_image)
    hsv_arr_edited = hsv_arr.copy()

    threshold = list(threshold)  # Copy the list
    threshold[0] = (threshold[0] / 2) % 180  # Because OpenCV has hue values only between 0..180

    for singleHsv in hsv_values:
        current_hue = int(singleHsv[0][0] / 2)
        lower_hue = (int(current_hue - threshold[0]) % 180)
        upper_hue = (int(current_hue + threshold[0]) % 180)

        current_sat = int(singleHsv[0][1])
        lower_sat = int(current_sat - threshold[1])
        upper_sat = int(current_sat + threshold[1])

        current_val = int(singleHsv[0][2])
        lower_val = int(current_val - threshold[2])
        upper_val = int(current_val + threshold[2])

        within_threshold_values_flag = np.zeros_like(hsv_arr[..., 0], dtype=bool)

        if upper_hue < lower_hue:
            # If hue value is at the wrapping point of the circle (0°/360°) apply threshold differently
            within_threshold_values_flag[
                (((hsv_arr[..., 0] > lower_hue) & (hsv_arr[..., 0] <= 180)) | (
                        (hsv_arr[..., 0] < upper_hue) & (hsv_arr[..., 0] >= 0))) &  # hue is within bounds
                (hsv_arr[..., 1] > lower_sat) & (hsv_arr[..., 1] < upper_sat) &  # saturation is within bounds
                (hsv_arr[..., 2] > lower_val) & (hsv_arr[..., 2] < upper_val)  # value is within bounds
                ] = True
        else:
            within_threshold_values_flag[
                (hsv_arr[..., 0] > lower_hue) & (hsv_arr[..., 0] < upper_hue) &  # hue is within bounds
                (hsv_arr[..., 1] > lower_sat) & (hsv_arr[..., 1] < upper_sat) &  # saturation is within bounds
                (hsv_arr[..., 2] > lower_val) & (hsv_arr[..., 2] < upper_val)  # value is within bounds
                ] = True

        if isinstance(singleHsv[1], str):
            index = index + 90

            # Rainbow effect: Increment hue by a specific value each frame
            if str(singleHsv[1]).startswith("Regenbogen"):
                params = str(singleHsv[1]).split(",")
                if len(params) != 2:
                    raise ValueError(
                        "The target hue for the rainbow effect must be passed as a string in the following "
                        "format: 'Regenbogen,5', where 5 is the value, the hue will be incremented in each "
                        "frame.")
                target_hue = int((current_hue + (index * int(params[1]))) / 2) % 180

            # Strobe effect: Increment the hue every n frames by a specific value
            elif str(singleHsv[1]).startswith("Strobe"):
                params = str(singleHsv[1]).split(",")
                if len(params) != 3:
                    raise ValueError(
                        "The target hue for the strobe effect must be passed as a string in the following "
                        "format: 'Strobe,10,30', where 10 is the number of frames the hue will stay the same, "
                        "and 30 the value, the hue will jump to.")
                target_hue = int((current_hue + (int(index / int(params[1])) * int(params[2]))) / 2) % 180
            else:
                raise ValueError("Target value " + str(singleHsv[1]) + " was not recognized.")

            hsv_arr_edited[within_threshold_values_flag, 0] = target_hue
        else:
            saturation_to_add = (singleHsv[1][1] - singleHsv[0][1])
            value_to_add = (singleHsv[1][2] - singleHsv[0][2])

            hsv_arr_clamped = hsv_arr.copy()

            # clamp saturation to prevent overflow
            hsv_arr_clamped[within_threshold_values_flag, 1] = np.clip(
                hsv_arr[within_threshold_values_flag, 1],
                max(0, saturation_to_add * -1),
                min(255, 255 - saturation_to_add))

            # clamp value to prevent overflow
            hsv_arr_clamped[within_threshold_values_flag, 2] = np.clip(
                hsv_arr[within_threshold_values_flag, 2],
                max(0, value_to_add * -1),
                min(255, 255 - value_to_add))

            hsv_arr_edited[within_threshold_values_flag, 0] = int(singleHsv[1][0] / 2)  # Sets the target hue
            hsv_arr_edited[within_threshold_values_flag, 1] = np.add(hsv_arr_clamped[within_threshold_values_flag, 1],
                                                                     saturation_to_add)  # Sets the target saturation
            hsv_arr_edited[within_threshold_values_flag, 2] = np.add(hsv_arr_clamped[within_threshold_values_flag, 2],
                                                                     value_to_add)  # Sets the target value

    bgr_result_img = cv2.cvtColor(hsv_arr_edited, cv2.COLOR_HSV2BGR)
    return bgr_result_img


def shift_hue_only(bgr_image, hues, threshold, index):
    """
    Shifts a hue on an image. This shifts hue only, so saturation and lightness of the color will not be changed.
    :param bgr_image: The image in BGR format
    :param hues: List of tuples (current hue [0..360], target hue [0..360 or string])
                 Possible string values for target hue:
                     'Regenbogen,n': Fade the colors in a rainbow, so for every single frame the hue will increment by n
                     'Strobe,n,m', Flash the colors, so for every n frames the hue will be incremented by m
    :type hues: List of tuples (int, int/string)
    :param threshold: How many hue values above and below the current hue will be shifted
    :param index: Index of the current frame. Is used for Regenbogen (rainbow) and Strobe effect
    :return: The shifted image
    """

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_arr = np.array(hsv_image)
    hsv_arr_edited = hsv_arr.copy()

    threshold = (threshold / 2) % 181  # Because OpenCV has hue values only between 0..180
    for single_hue in hues:
        current_hue = int(single_hue[0] / 2)
        lower_hue = (int(current_hue - threshold) % 181)
        upper_hue = (int(current_hue + threshold) % 181)

        # Rainbow effect: Increment hue by a specific value each frame
        if str(single_hue[1]).startswith("Regenbogen"):
            params = str(single_hue[1]).split(",")
            if len(params) != 2:
                raise ValueError("The target hue for the rainbow effect must be passed as a string in the following "
                                 "format: 'Regenbogen,5', where 5 is the value, the hue will be incremented in each "
                                 "frame.")
            target_hue = int((single_hue[0] + (index * int(params[1]))) / 2)

        # Strobe effect: Increment the hue every n frames by a specific value
        elif str(single_hue[1]).startswith("Strobe"):
            params = str(single_hue[1]).split(",")
            if len(params) != 3:
                raise ValueError("The target hue for the strobe effect must be passed as a string in the following "
                                 "format: 'Strobe,10,30', where 10 is the number of frames the hue will stay the same, "
                                 "and 30 the value, the hue will jump to.")
            target_hue = int((single_hue[0] + (int(index / int(params[1])) * int(params[2]))) / 2)
        else:
            target_hue = int(single_hue[1] / 2)

        if upper_hue < lower_hue:
            # If hue value is at the wrapping point of the circle (0°/360°) apply threshold differently
            hsv_arr_edited[..., 0][((hsv_arr[..., 0] > lower_hue) & (hsv_arr[..., 0] <= 180)) | (
                    (hsv_arr[..., 0] < upper_hue) & (hsv_arr[..., 0] >= 0))] = target_hue
        else:
            hsv_arr_edited[..., 0][(hsv_arr[..., 0] > lower_hue) & (hsv_arr[..., 0] < upper_hue)] = target_hue

    result_bgr_img = cv2.cvtColor(hsv_arr_edited, cv2.COLOR_HSV2BGR)
    return result_bgr_img
