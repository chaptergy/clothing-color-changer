from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import cv2
import time
import numpy as np
import traceback

import ui_mainwindow
import ui_advanced
import imageProcessor as ip
import progressHandler
import colorNameConverter

fileIsImage = False
workThread = None
mainW = None

colors = np.empty((0, 2, 3), dtype=np.uint8)  # Saves the colors to shift (HSV)
overlayColorRGB = np.array([255, 255, 0])
thresholds = [30, 60, 180]
feather = 10
shiftInBox = False

showMasks = False
showBox = False
showGroups = False

groupClusterSize = 4
groupPrecision = 3


class mainWindowClass(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        global overlayColorRGB
        QtWidgets.QWidget.__init__(self, parent)
        self.oriPixmap = None
        self.resPixmap = None
        self.ui = ui_mainwindow.Ui_mainWindow()
        self.ui.setupUi(self)
        self.installEventFilter(self)
        self.currentSelColor = None
        self.ui.colorSelect_mask.setStyleSheet("background-color:rgb(" + ",".join(overlayColorRGB.astype(str)) + ")")

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.Resize):
            # Scale Pixmaps when window is resized
            if self.oriPixmap is not None:
                self.ui.origImgLabel.setPixmap(
                    self.oriPixmap.scaled(self.ui.origImgLabel.size(), QtCore.Qt.KeepAspectRatio,
                                          QtCore.Qt.SmoothTransformation))
            if self.resPixmap is not None:
                self.ui.resultImgLabel.setPixmap(
                    self.resPixmap.scaled(self.ui.resultImgLabel.size(), QtCore.Qt.KeepAspectRatio,
                                          QtCore.Qt.SmoothTransformation))
        return super(mainWindowClass, self).eventFilter(source, event)

    def select_new_frame(self):
        pass

    def add_color(self):
        global colors
        new_index = len(colors)

        self.ui.colorsListView.addItem("Rot zu Rot")
        new = np.array([[[0, 255, 255], [0, 255, 255]]])
        colors = np.append(colors, new, axis=0)
        self.ui.colorsListView.setCurrentRow(new_index)
        self.currentSelColor = new_index
        self.ui.colorSelect_currentColor.setStyleSheet("background-color:hsv(0,255,255)")
        self.ui.colorSelect_destColor.setStyleSheet("background-color:hsv(0,255,255)")
        self.ui.destHue_color.setChecked(True)
        self.ui.colorSelect_currentColor.setEnabled(True)
        self.ui.colorSelect_destColor.setEnabled(True)
        self.show_dest_color_options(True)

    def remove_color(self):
        global colors
        self.ui.colorsListView.takeItem(self.currentSelColor)
        colors = np.delete(colors, self.currentSelColor, axis=0)
        self.ui.colorsListView.setCurrentRow(self.currentSelColor)
        try:
            self.change_selected_color(self.currentSelColor)
        except IndexError:
            try:
                self.change_selected_color(self.currentSelColor - 1)
            except IndexError:
                self.currentSelColor = None
                self.show_dest_color_options(True)
                self.ui.colorSelect_currentColor.setEnabled(False)
                self.ui.colorSelect_destColor.setEnabled(False)
                self.ui.colorSelect_currentColor.setStyleSheet("")
                self.ui.colorSelect_destColor.setStyleSheet("")

    def save_file(self):
        global fileIsImage
        if fileIsImage:
            file_filter = "Image files (*.jpg *.jpeg *.png *.bmp)"
        else:
            file_filter = "Video file (*.avi *.mp4)"
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', filter=file_filter)
        if not file_name or (file_name[0] == ""):
            return
        if fileIsImage:
            workThread.push(ip.save_image_to_file, [file_name[0]])
        else:
            workThread.push(ip.save_video_to_file, [file_name[0]])

    def show_advanced_options(self):
        adv_dialog = advancedWindowClass(self)
        adv_dialog.exec_()

    @staticmethod
    def set_show_masks(show):
        global showMasks
        showMasks = show

    @staticmethod
    def set_show_box(show):
        global showBox
        showBox = show

    @staticmethod
    def set_group_masks(show):
        global showGroups
        showGroups = show

    def set_target_solid_color(self):
        if colors[self.currentSelColor, 1, 0] < 0:  # nur wenn es vorher anders war
            self.ui.colorSelect_destColor.setEnabled(True)
            colors[self.currentSelColor, 1] = [0, 255, 255]
            self.ui.colorSelect_destColor.setStyleSheet("background-color:hsv(0,255,255)")
        self.show_dest_color_options(True)
        self.ui.colorsListView.item(self.currentSelColor).setText(self.generate_item_name(self.currentSelColor))

    def set_target_rainbow(self):
        self.ui.colorSelect_destColor.setEnabled(False)
        colors[self.currentSelColor, 1] = [-1, -1, 5]
        self.ui.colorchangeSpeedSlider.setSliderPosition(5)
        self.ui.colorSelect_destColor.setStyleSheet("")
        self.show_dest_color_options(False)
        self.ui.colorsListView.item(self.currentSelColor).setText(self.generate_item_name(self.currentSelColor))

    def set_target_flash(self):
        self.ui.colorSelect_destColor.setEnabled(False)
        colors[self.currentSelColor, 1] = [-2, -2, 5]
        self.ui.colorchangeSpeedSlider.setSliderPosition(5)
        self.ui.colorSelect_destColor.setStyleSheet("")
        self.show_dest_color_options(False)
        self.ui.colorsListView.item(self.currentSelColor).setText(self.generate_item_name(self.currentSelColor))

    def change_color_speed(self, speed):
        global colors
        colors[self.currentSelColor, 1, 2] = speed

    @staticmethod
    def generate_item_name(index):
        global colors
        current_color_name = colorNameConverter.hsv_to_name((colors[index, 0, 0] / 2, colors[index, 0, 1], colors[index, 0, 2]))

        if colors[index, 1, 0] == -1:
            target_color_name = "Farbwechsel"  # Rainbow
        elif colors[index, 1, 0] == -2:
            target_color_name = "Blinken"  # Strobe
        else:
            target_color_name = colorNameConverter.hsv_to_name(
                (colors[index, 1, 0] / 2, colors[index, 1, 1], colors[index, 1, 2]))
        return current_color_name + " zu " + target_color_name

    def set_color_current(self):
        global colors
        color = QtWidgets.QColorDialog.getColor()
        new_hsv = np.array(color.getHsv())[:3]
        self.ui.colorSelect_currentColor.setStyleSheet("background-color:hsv(" + ",".join(new_hsv.astype(str)) + ")")
        colors[self.currentSelColor, 0] = new_hsv
        self.ui.colorsListView.item(self.currentSelColor).setText(self.generate_item_name(self.currentSelColor))

    def set_color_target(self):
        global colors
        color = QtWidgets.QColorDialog.getColor()
        new_hsv = np.array(color.getHsv())[:3]
        self.ui.colorSelect_destColor.setStyleSheet("background-color:hsv(" + ",".join(new_hsv.astype(str)) + ")")
        colors[self.currentSelColor, 1] = new_hsv
        self.ui.colorsListView.item(self.currentSelColor).setText(self.generate_item_name(self.currentSelColor))

    def set_color_overlay(self):
        global overlayColorRGB
        color = QtWidgets.QColorDialog.getColor()
        new_rgb = np.array(color.getRgb())[:3]
        self.ui.colorSelect_mask.setStyleSheet("background-color:rgb(" + ",".join(new_rgb.astype(str)) + ")")
        overlayColorRGB = new_rgb

    def show_dest_color_options(self, isSolid):
        self.ui.label_5.setVisible(not isSolid)
        self.ui.colorchangeSpeedSlider.setVisible(not isSolid)
        self.ui.label_4.setVisible(isSolid)
        self.ui.colorSelect_destColor.setVisible(isSolid)

    def color_option_selected(self, selectedObject):
        self.change_selected_color(selectedObject.row())

    def change_selected_color(self, index):
        global colors
        self.currentSelColor = index
        if colors[index, 1, 0] == -1:
            self.ui.destHue_rainbow.setChecked(True)
            self.show_dest_color_options(False)
        elif colors[index, 1, 0] == -2:
            self.ui.destHue_strobe.setChecked(True)
            self.show_dest_color_options(False)
        else:
            self.ui.destHue_color.setChecked(True)
            self.ui.colorSelect_destColor.setEnabled(True)
            self.show_dest_color_options(True)
            self.ui.colorSelect_destColor.setStyleSheet(
                "background-color:hsv(" + ",".join(colors[index, 1].astype(str)) + ")")
        self.ui.colorSelect_currentColor.setEnabled(True)
        self.ui.colorSelect_currentColor.setStyleSheet(
            "background-color:hsv(" + ",".join(colors[index, 0].astype(str)) + ")")

    def process_data(self):
        global colors, thresholds, overlayColorRGB, showMasks, showBox, showGroups, feather, \
            shiftInBox, groupClusterSize, groupPrecision
        values_to_be_processed = colors.tolist()
        for i in range(0, len(values_to_be_processed)):
            if values_to_be_processed[i][1][0] == -1:
                values_to_be_processed[i][1] = "Regenbogen," + str(int(values_to_be_processed[i][1][2]))
            elif values_to_be_processed[i][1][0] == -2:
                values_to_be_processed[i][1] = "Strobe," + str(int(values_to_be_processed[i][1][2])) + ",20"
        workThread.push(ip.process_all_images,
                        [values_to_be_processed.copy(), thresholds.copy(), feather,
                         'masks' if not shiftInBox else 'boxes',
                         showMasks, showBox, showGroups, overlayColorRGB[::-1].tolist(), groupClusterSize,
                         groupPrecision])


class advancedWindowClass(QtWidgets.QDialog):
    def __init__(self, parent=None):
        global thresholds, feather, shiftInBox, groupClusterSize, groupPrecision
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_advanced.Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.spin_hue.setProperty("value", thresholds[0])
        self.ui.spin_sat.setProperty("value", thresholds[1])
        self.ui.spin_val.setProperty("value", thresholds[2])
        self.ui.spin_cluster.setProperty("value", groupClusterSize)
        self.ui.spin_precision.setProperty("value", groupPrecision)
        self.ui.slider_feather.setProperty("value", feather)
        self.ui.checkBox_shiftInBox.setChecked(shiftInBox)

    @staticmethod
    def set_threshold_hue(val):
        global thresholds
        thresholds[0] = val

    @staticmethod
    def set_threshold_sat(val):
        global thresholds
        thresholds[1] = val

    @staticmethod
    def set_threshold_val(val):
        global thresholds
        thresholds[2] = val

    @staticmethod
    def set_feather_radius(val):
        global feather
        feather = val

    @staticmethod
    def set_shift_hitbox(val):
        global shiftInBox
        shiftInBox = val

    @staticmethod
    def set_cluster_size(val):
        global groupClusterSize
        groupClusterSize = val

    @staticmethod
    def set_group_precision(val):
        global groupPrecision
        groupPrecision = val


class mainThread(QtCore.QThread):
    '''
    Runs all functions in a thread separate from the ui thread
    '''

    startProgress = QtCore.pyqtSignal(str, bool)
    notifyProgress = QtCore.pyqtSignal(int)
    endProgress = QtCore.pyqtSignal()
    notifyError = QtCore.pyqtSignal(str)
    notifyDone = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.queue = []
        self.nap_time = .1
        self.n = 0
        self.serving = None
        self.return_values = {}
        progressHandler.thread = self

    def error(self, error, function, a=(), kw=None):
        print('An error occured (', function, a, kw or {}, '):', error)
        traceback.print_exc()
        self.notifyError.emit(
            'An error occured (' + str(function) + str(a) + str(kw) or str({}) + '):' + str(error))

    def push(self, function, a=(), kw=None):
        ''' queue the call to function, return the number of the call. '''
        self.queue.append((function, a, kw or {}))
        return self.n + len(self.queue)

    def push_and_wait(self, function, a=(), kw=None):
        ''' queue the call to function, return the returned value. '''
        self.push(function, a, kw or {})
        return self.get(self.n + len(self.queue))

    def stop(self):
        self.running = False

    def get(self, n):
        ''' Block until self.n >= n. '''
        while (self.n < n) | (self.serving is not None):
            time.sleep(self.nap_time)
        return self.return_values.get(n, None)

    def run(self):
        self.running = True
        while self.running:
            if len(self.queue) == 0:
                time.sleep(self.nap_time)
            else:
                function, a, kw = self.queue.pop(0)
                self.serving = (function, a, kw)
                self.n += 1
                try:
                    self.return_values[self.n] = function(*a, **kw)
                    if (function == ip.process_all_images) | (function == ip.run_masked_hue_shift):
                        self.notifyDone.emit()
                except Exception as error:
                    self.error(error, function, a, kw)
                self.serving = None


def CVtoQImage(im, copy=False):
    if im is None:
        return QtGui.QImage()

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
    return qim.copy() if copy else qim


def showImage(img, isOrig=False):
    global mainW

    convImg = CVtoQImage(img)  # QImage object
    if isOrig:
        mainW.oriPixmap = QtGui.QPixmap.fromImage(convImg)
    else:
        mainW.resPixmap = QtGui.QPixmap.fromImage(convImg)
    mainW.eventFilter(None, QtCore.QEvent(QtCore.QEvent.Resize))


def startProgress(message=None, marquee=False):
    global mainW
    mainW.ui.statusProgressBar.setValue(0)
    if marquee:
        mainW.ui.statusProgressBar.setRange(0, 0)
    else:
        mainW.ui.statusProgressBar.setRange(0, 1)
    mainW.ui.statusProgressBar.show()

    if message is not None:
        mainW.ui.statusbar.showMessage(message)
    else:
        mainW.ui.statusbar.showMessage("")


def changeProgress(progress):
    global mainW
    mainW.ui.statusProgressBar.setValue(int(progress))


def endProgress():
    global mainW
    mainW.ui.statusProgressBar.hide()
    mainW.ui.statusProgressBar.setValue(0)
    mainW.ui.statusProgressBar.setRange(0, 1)
    mainW.ui.statusbar.showMessage("")


def showMessage(message, isError=True):
    msg = QtWidgets.QMessageBox()
    msg.setText(message)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

    if isError:
        msg.setWindowTitle("Error")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
    else:
        msg.setWindowTitle("Info")
        msg.setIcon(QtWidgets.QMessageBox.Information)

    msg.exec_()


def processingDone():
    showImage(workThread.pushAndWait(ip.get_image_at_frame, [0]), isOrig=False)


def main():
    global workThread, mainW, fileIsImage

    workThread = mainThread()
    workThread.startProgress.connect(startProgress)
    workThread.notifyProgress.connect(changeProgress)
    workThread.endProgress.connect(endProgress)
    workThread.notifyError.connect(showMessage)
    workThread.notifyDone.connect(processingDone)
    workThread.start()
    app = QtWidgets.QApplication(sys.argv)
    mainW = mainWindowClass()
    fileValid = False
    while not fileValid:
        fname = QtWidgets.QFileDialog.getOpenFileName(mainW, 'Open file', 'C:/',
                                                      'Image files (*.jpg *.jpeg *.bmp);;Video files (*.avi *.mp4)')
        if not fname or (fname[0] == ""):
            return
        if (fname[0].endswith(".jpg")) | (fname[0].endswith(".jpeg")) | (fname[0].endswith(".bmp")):
            fileIsImage = True
        elif (fname[0].endswith(".avi")) | (fname[0].endswith(".mp4")):
            fileIsImage = False
        else:
            showMessage("The file must be one of the following filetypes:\n.jpg  .jpeg  .bmp  .avi  .mp4")
            continue

        fileValid = True

        if fileIsImage:
            workThread.push(ip.load_image_from_file, [fname[0]])
        else:
            workThread.push(ip.load_video_from_file, [fname[0]])

    mainW.show()
    orImg = workThread.push_and_wait(ip.get_original_image_at_frame, [0])
    showImage(orImg, isOrig=True)
    #    if fileIsImage:
    #        mainW.ui.btnSelectFrame.setEnabled(False)
    #        mainW.ui.currentFrameLabel.setEnabled(False)
    #        mainW.ui.currentFrameLabel.setText("")
    #    else:
    #        mainW.ui.btnSelectFrame.setEnabled(True)
    #        mainW.ui.currentFrameLabel.setEnabled(True)
    #        mainW.ui.currentFrameLabel.setText("Aktueller Frame: 1")

    #    workThread.push(ip.runMaskRCNN)

    sys.exit(app.exec_())
    exit(0)


main()
