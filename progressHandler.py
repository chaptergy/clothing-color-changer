import sys

thread = None
progress_length = 30  # Length of progressbar in console


def start_progress(message=None, marquee=False):
    """
    Show a progressbar with 0%
    :param message: Message next to the progress bar
    :param marquee: Whether the bar should be a marquee
    """
    '''
    Fortschritleiste mit 0% anzeigen.
    '''
    global thread, progress_length
    if message is not None:
        print(message)
    if not marquee:
        sys.stdout.write("Progress: │{0}│ {1}%".format(' ' * progress_length, 0))
        sys.stdout.flush()

    thread.startProgress.emit(message, marquee)


def progress(percent):
    """
    Update progress for progressbar
    :param percent: How far the progress bar should be (e.g. 0.15 for 15%)
    :type percent: float
    """

    global thread, progress_length

    arrow = '░' * int(round(percent * progress_length))
    spaces = ' ' * (progress_length - len(arrow))
    sys.stdout.write("\rProgress: │{0}│ {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

    thread.notifyProgress.emit(percent)


def end_progress():
    """
    Remove progressbar
    """
    global thread, progress_length

    sys.stdout.write("\rProgress: │{0}│ {1}%\n".format('░' * progress_length, 100))
    sys.stdout.flush()

    thread.endProgress.emit()
