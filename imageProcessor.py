import os
import cv2
import numpy
import traceback
import timeit
import gzip

try:
    import ujson as json
except ImportError:
    import json
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

import videoHelper as vh
import visualizePersonMasks as visualize
import hueShifter as hS
import groupMasks
import progressHandler as progress
import myLogger as log
import coco

# Root directory of Mask RCNN
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Variables for RCNN
config = None
model = None

initialized = False
openedFileName = ""

# Container for Image- and RCNN-data
images = []
mrcnn_res = {}
resultImages = {}

log.debug('#' * 60)


def init():
    """
    Initialize Mask RCNN
    """
    global model, MODEL_DIR, config, COCO_MODEL_PATH, ROOT_DIR, initialized

    start_time = timeit.default_timer()

    print("Pre import")

    import model as modellib

    print("Import done")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    print("Set config")

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    print("set model")

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    print("loaded weights")

    initialized = True

    elapsed_time = timeit.default_timer() - start_time
    log.info("Mask RCNN was initialized in", round(elapsed_time, 5), "seconds.")


def load_image_from_file(path, max_image_size=(1280, 720)):
    """
    Loads an image from file into the images array and scales it down if necessary
    :param path: Path to the image
    :param max_image_size: The maximum size the image should have. To disable resizing use None
    :type max_image_size: (int width, int height) or None
    """
    global images, mrcnn_res, openedFileName

    progress.start_progress("Loading image...", marquee=True)

    mrcnn_res.clear()  # Clear M-RCNN result array
    images.clear()  # Clear image array to remove all previously loaded images

    loaded_image = cv2.imread(path)

    if loaded_image is not None:  # If image has been loaded successfully
        images.insert(0, loaded_image)
        # If image is too large, resize it
        if (max_image_size is not None) & \
                ((images[0].shape[0] > max_image_size[0]) | (images[0].shape[1] > max_image_size[1])):
            percent = min(max_image_size[0] / images[0].shape[0], max_image_size[1] / images[0].shape[1])
            images[0] = cv2.resize(images[0], None, fx=percent, fy=percent)

        openedFileName = os.path.basename(path)

        log.info("File", path, "has been loaded.")
        progress.end_progress()
    else:
        progress.end_progress()
        raise IOError('An error occurred while loading the image!')


def save_image_to_file(path):
    """
    Saves a processed image to a path
    :param path: Path to the new image
    """
    global resultImages, images

    progress.start_progress("Saving image...", marquee=True)

    if 0 not in resultImages:
        log.warning("There is no processed frame yet!")
    else:
        cv2.imwrite(path, resultImages[0])
        os.startfile(path)
        log.info("File", path, "was saved.")
    progress.end_progress()


def load_video_from_file(path, max_fps=20, max_image_size=(1280, 720)):
    """
    Loads a video from file, reduces framerate and scales it down if necessary
    :param path: Path to the video
    :param max_fps: Maximum frames per second for the output file
    :param max_image_size: The maximum size the video should have. To disable resizing use None
    :type max_image_size: (int width, int height) or None
    """
    global images, mrcnn_res, openedFileName

    progress.start_progress("Loading video...", marquee=True)

    mrcnn_res.clear()  # Clear M-RCNN result array
    images.clear()  # Clear image array to remove all previously loaded images

    images = vh.video_to_images(path, max_fps, max_image_size)

    openedFileName = os.path.basename(path)

    log.info("File", path, "has been loaded.")

    progress.end_progress()


def save_video_to_file(path, fps=20):
    """
    Saves the result video to a file
    :param path: Path to the new video
    :param fps: Frames per second of the new file
    """
    global resultImages

    progress.start_progress("Saving video...", marquee=True)

    if len(resultImages) <= 0:
        log.warning("There are no processed frames yet!")
    else:
        img_values = [v for v in resultImages.values()]  # Convert dictionary to array
        vh.images_to_video(img_values, path, fps)

    progress.end_progress()


def run_mask_rcnn():
    """
    Performs the mask rcnn for all images in the 'images' array.
    The results will be saved into 'mrcnn_res' at the same index as the image.
    """

    global images, mrcnn_res, initialized, openedFileName, ROOT_DIR

    debug_file_path = ROOT_DIR + '\\debug\\' + openedFileName + '_maskRCNN-Data.json.gz'

    # Mrcnn data has already been calculated
    if len(mrcnn_res) > 0:
        return

    start_time = timeit.default_timer()
    try:
        if os.path.isfile(debug_file_path):  # If file with mrcnn data exists, load it instead of generating it anew
            with gzip.open(debug_file_path, 'rt') as infile:
                progress.start_progress("Loading persons data...", marquee=True)
                log.info("Mask RCNN data is loaded from file...")
                in_data = dict(json.load(infile))
                infile.close()
                mrcnn_res = {int(k): {k2: numpy.array(v2) for k2, v2 in v.items()} for k, v in in_data.items()}
                elapsed_time = timeit.default_timer() - start_time
                log.info(len(images), "images(s) and their Mask RCNN data were loaded in",
                         round(elapsed_time, 5), "Seconds.")
                progress.end_progress()
                return

    except Exception:
        print("Mask RCNN Data could not be loaded. The file seems to be corrupted:")
        traceback.print_exc()
        os.remove(debug_file_path)

    # If Mask RCNN has not yet been initialized, initialize it
    if not initialized:
        log.info("Initializing Mask RCNN...")
        init()
        start_time = timeit.default_timer()

    nr_of_images_to_process = len(images)
    log.info("Starting to process ", nr_of_images_to_process, "image(s) with Mask RCNN...")
    progress.start_progress("Personen werden erkannt...")
    for index, img in enumerate(images):
        img_res = model.detect([img])  # Run object detection

        # Filter masks, which show persons
        is_person_mask = (img_res[0]['class_ids'][:] == 1)

        # Remove all other masks
        img_res[0]['class_ids'] = img_res[0]['class_ids'][is_person_mask]
        img_res[0]['rois'] = img_res[0]['rois'][is_person_mask]
        img_res[0]['scores'] = img_res[0]['scores'][is_person_mask]
        img_res[0]['masks'] = img_res[0]['masks'][:, :, is_person_mask]

        # Save the result in an array
        mrcnn_res[index] = img_res[0]

        progress.progress((index + 1) / nr_of_images_to_process)

    progress.end_progress()

    elapsed_time = timeit.default_timer() - start_time
    log.info(len(images), "images(s) were processed with Mask RCNN in", round(elapsed_time, 5),
             "Seconds.")
    start_time = timeit.default_timer()

    # If doesn't exist, save to file for later reuse
    if not os.path.isfile(debug_file_path):
        progress.start_progress("Saving persons data...", marquee=True)
        try:
            out_data = mrcnn_res.copy()
            log.info("Saving Mask RCNN data to file...")
            out_data = {k: {k2: v2.tolist() for k2, v2 in v.items()} for k, v in out_data.items()}
            with gzip.open(debug_file_path, 'wt') as outfile:
                json.dump(out_data, outfile)
                outfile.close()

        except Exception:
            print("Could not save Mask RCNN data:")
            traceback.print_exc()
        progress.end_progress()

    elapsed_time = timeit.default_timer() - start_time
    log.info("Mask-data of", len(images), "images(s) have been to file in", round(elapsed_time, 5), "Seconds.")


def run_masked_hue_shift(hsv_values=None, threshold=(30, 60, 180), feather=0, shift_in='masks', draw_mask=False,
                         draw_box=False, draw_groups=False, bgr_draw_color=(0, 255, 255), nr_of_color_groups=4,
                         grouping_precision=3):
    """
    Runs a hue shift in each mask. If mask rcnn has not been called before, ot will call it.
    :param hsv_values: List of lists (current hsv ([0..360],[0..255],[0..255]), target hue ([0..360],[0..255],[0..255]))
                 The target hue tuple can be replaced by one of the following string values:
                     'Regenbogen,n': Fade the colors in a rainbow, so for every single frame the hue will increment by n
                     'Strobe,n,m', Flash the colors, so for every n frames the hue will be incremented by m
    :type hsv_values: List of lists containing ((int, int, int),(int, int, int)/string))
    :param threshold: How many hue values above and below the current hue, saturation and value will be shifted.
                      The best results are when hue is low, saturation medium and value high (e.g. [30,60,160])
    :type threshold: (int, int, int)
    :param feather: The amount of feathering applied around the mask
    :param shift_in: Where to shift the colors. Can be 'masks' (detailed masks) or 'boxes' (bounding boxes)
    :param draw_mask: Whether to draw the masks on the final image
    :param draw_box: Whether to draw the bounding boxes on the final image
    :param draw_groups: Whether to print the group id near each mask
    :param bgr_draw_color: The color of all the drawn elements in BGR format
    :param nr_of_color_groups: Into how many groups the colors in each mask should be grouped
    :param grouping_precision: he precision with which the masks will be put into groups (the larger the precision,
    the more similar the colors have to be to be grouped into the same group)
    """
    global images, mrcnn_res, resultImages

    start_time = timeit.default_timer()

    nr_of_images_to_process = len(images)
    log.info("Begin processing", nr_of_images_to_process, "image(s) with hue shift...")

    progress.start_progress("Running hue shifts...")
    for index, img in enumerate(images):
        log.debug("Processing image ", index + 1, "...")

        img = img.copy()
        orig_img = img.copy()

        # No rcnn result available
        if index not in mrcnn_res:
            progress.end_progress()
            log.info("Mask RCNN has not been run for frame " + str(index))
            run_mask_rcnn()
            progress.start_progress("Running hue shifts...")

        # Still no RCNN result available
        if index not in mrcnn_res:
            raise RuntimeError("Unable to generate Mask RCNN Data for frame " + str(index) + " !")

        r = mrcnn_res[index]

        if len(r['rois']) > 0:
            if hsv_values is not None:
                shifted_image = hS.shift_hsv(img, hsv_values, threshold, index)

                progress.progress(((index * 3) + 1) / (nr_of_images_to_process * 3))

                # If color changing should be done in entire bounding box
                if shift_in == 'boxes':
                    for i, box in enumerate(r['rois']):  # overwrite each pixel in bounding box with shifted image
                        img[box[0]:box[2], box[1]:box[3]] = shifted_image[box[0]:box[2], box[1]:box[3]]

                # Otherwise change color in mask only
                else:
                    bin_mask = numpy.amax(r['masks'], axis=-1)
                    if feather > 0:
                        bin_mask = bin_mask.astype(numpy.float)
                        bin_mask = binary_dilation(bin_mask, iterations=feather).astype(
                            numpy.float)  # Dialate mask, so it really contains everything
                        numpy.array(gaussian_filter(bin_mask, sigma=feather, output=bin_mask))  # Feather masks

                    img[..., 0] = img[..., 0] * (1 - bin_mask[...]) + shifted_image[..., 0] * bin_mask[...]  # blue
                    img[..., 1] = img[..., 1] * (1 - bin_mask[...]) + shifted_image[..., 1] * bin_mask[...]  # green
                    img[..., 2] = img[..., 2] * (1 - bin_mask[...]) + shifted_image[..., 2] * bin_mask[...]  # red

            progress.progress(((index * 3) + 2) / (nr_of_images_to_process * 3))

            in_groups = None
            out_groups = None
            if draw_groups:
                in_groups, in_group_info = groupMasks.group_masks_by_dyn_color(orig_img, r, inside=True,
                                                                               nr_of_color_groups=nr_of_color_groups,
                                                                               grouping_precision=grouping_precision)
                out_groups, out_group_info = groupMasks.group_masks_by_dyn_color(orig_img, r, inside=False,
                                                                                 nr_of_color_groups=nr_of_color_groups,
                                                                                 grouping_precision=grouping_precision)
                log.debug("Image " + str(index + 1) + ":\n Groups by mask:\n" +
                          str(groupMasks.hue_array_to_string_array(in_group_info)) + "\n Groups by bounding boxes:\n" +
                          str(groupMasks.hue_array_to_string_array(out_group_info)))

            # Only if mask, box or group number should be drawn onto the image
            if draw_mask | draw_box | draw_groups:
                img = visualize.draw_masks_on_image(img, r['rois'], r['masks'], in_groups, out_groups, draw_groups,
                                                    draw_mask, draw_box, bgr_draw_color)

        else:
            log.debug("On image ", index, "no people have been recognized.")

        resultImages[index] = img

        progress.progress(((index * 3) + 3) / (nr_of_images_to_process * 3))

    progress.end_progress()

    elapsed_time = timeit.default_timer() - start_time
    log.info(len(images), "image(s) have been processed by hue shift in", round(elapsed_time, 5), "seconds.")


def process_all_images(hsv_values, threshold=(30, 20, 20), feather=0, shift_in='masks', draw_mask=False, draw_box=False,
                       draw_groups=False, bgr_draw_color=(0, 255, 255), nr_of_color_groups=4, grouping_precision=3):
    """
    Runs mask rcnn first, and then hue shifts all images
     :param hsv_values: List of lists (current hsv ([0..360],[0..255],[0..255]), target hue ([0..360],[0..255],[0..255]))
                 The target hue tuple can be replaced by one of the following string values:
                     'Regenbogen,n': Fade the colors in a rainbow, so for every single frame the hue will increment by n
                     'Strobe,n,m', Flash the colors, so for every n frames the hue will be incremented by m
    :type hsv_values: List of lists containing ((int, int, int),(int, int, int)/string))
    :param threshold: How many hue values above and below the current hue, saturation and value will be shifted.
                      The best results are when hue is low, saturation medium and value high (e.g. [30,60,160])
    :type threshold: (int, int, int)
    :param feather: The amount of feathering applied around the mask
    :param shift_in: Where to shift the colors. Can be 'masks' (detailed masks) or 'boxes' (bounding boxes)
    :param draw_mask: Whether to draw the masks on the final image
    :param draw_box: Whether to draw the bounding boxes on the final image
    :param draw_groups: Whether to print the group id near each mask
    :param bgr_draw_color: The color of all the drawn elements in BGR format
    :param nr_of_color_groups: Into how many groups the colors in each mask should be grouped
    :param grouping_precision: he precision with which the masks will be put into groups (the larger the precision,
    the more similar the colors have to be to be grouped into the same group)
    """

    run_mask_rcnn()
    run_masked_hue_shift(hsv_values, threshold, feather, shift_in, draw_mask, draw_box, draw_groups, bgr_draw_color, nr_of_color_groups,
                         grouping_precision)


def get_image_at_frame(frame=0):
    """
    Returns the processed frame at a specified index
    :param frame: Index of the frame to return
    :return: The processed image
    """

    global resultImages

    if (len(resultImages) > 0) & (frame < len(resultImages)):
        return resultImages[frame]
    else:
        return None


def get_original_image_at_frame(frame=0):
    """
    Returns the unprocessed image at a specified index
    :param frame: Index of the frame to return
    :return: The unprocessed image
    """

    global images

    if (len(images) > 0) & (frame < len(images)):
        return images[frame]
    else:
        return None
