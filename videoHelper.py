import numpy as np
import cv2
import os.path
import warnings
import myLogger as log


def video_to_images(video_path, max_fps=20, max_size=None):
    """
    Converts a video into a list of images and returns it. If necessary, it lowers the framerate and image size.
    :param video_path: Path to the video
    :param max_fps: Maximum frames per second for the output file
    :param max_size: The maximum size the video should have. To disable resizing use None
    :type max_size: (int width, int height) or None
    :return: List of images
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError('Error loading the video file!')

    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    frames_to_drop = []
    if (max_fps > 0) & (fps > max_fps):
        nr_of_frames_to_be_removed = fps - max_fps

        # calculate, which frames should be dropped (in regular intervals)
        frames_to_drop = np.linspace(-1, fps - 1, nr_of_frames_to_be_removed + 1, endpoint=True).round()
        frames_to_drop = frames_to_drop[1:]  # remove first element (always -1)

    log.info(len(frames_to_drop), "frames are dropped per second, to reduce the fps from", fps, "to", max_fps)

    img_list = []

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            # reduce framerate
            frame_counter = (frame_counter + 1)
            if np.in1d((frame_counter % (fps + 1)), frames_to_drop):  # drop frame if in drop array
                continue

            # shrink image
            if (max_size is not None) & ((frame.shape[0] > max_size[0]) | (frame.shape[1] > max_size[1])):
                perc = min(max_size[0] / frame.shape[0], max_size[1] / frame.shape[1])
                frame = cv2.resize(frame, None, fx=perc, fy=perc)

            img_list.append(frame)
        else:
            break

    cap.release()
    return img_list


def images_to_video(img_list, destination_path='output.avi', fps=20.):
    """
    Converts a list of images to a video and saves it to file
    :param img_list: List of images
    :param destination_path: Path to the file where the video should be saved
    :param fps: The frames per second of the output video
    """
    try:
        # If file already exists, delete it
        os.remove(destination_path)
    except OSError:
        pass

    filename, file_extension = os.path.splitext(destination_path)
    if file_extension != '.avi':
        warnings.warn("When the file extension is not .avi, it may happen, that no file will be saved!")
    height, width, channels = img_list[0].shape  # frameSize rausfinden

    #  VideoWriter needs to have the right codec depending on the system, so try multiple fourcc codecs
    counter = 0
    fourcc_array = ['X264', 'XVID', 'DIVX', 'MJPG', 'MRLE', 'Custom']
    while counter < len(fourcc_array):  # Try to find a working codec
        if fourcc_array[counter] != 'Custom':
            fourcc = cv2.VideoWriter_fourcc(*fourcc_array[counter])
        else:
            # When setting fourcc to -1, a dialog will show at runtime
            # allowing the user to select one of the availabe codecs
            fourcc = -1
        out = cv2.VideoWriter(destination_path, fourcc, fps, (width, height), True)
        for img in img_list:
            out.write(np.uint8(img))

        out.release()
        try:
            # If file was saved successfully, so file size is larger than 5 bytes
            if os.path.getsize(destination_path) > 5:
                if counter > 0:
                    log.debug("Saving with codec(s)", ", ".join([item for item in fourcc_array[:counter]]),
                              "failed.")
                log.info("File", destination_path, "was saved with codec", fourcc_array[counter])
                return
        except:
            pass
        counter += 1
    raise Exception("Unable to save" + str(destination_path) + "!")
