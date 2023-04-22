from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QDialog, QLabel, QLineEdit
from PyQt5 import QtCore, QtGui, QtWidgets
from threading import Thread
import base64
import json
import pandas as pd
import cv2
import imutils
import time
from datetime import datetime
from collections import deque
import sys
import requests
from multiprocessing import Pool
import numpy as np
import datetime


class YOLO:
    def __init__(self, url = 'http://127.0.0.1:5000/predict'):
        self.url = 'http://10.250.1.94:5000/predict'
        #self.url = url
        self.do_ReID = False
        self.camId = None
        self.new_bounding_box_cord = None
        self.length = None
        self.session = requests.Session()

    def scale_bbox(self,bbox, original_size, resized_size):
        x, y, w, h = bbox
        original_height, original_width = original_size
        resized_height, resized_width = resized_size

        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        return [x, y, w, h]

    def get_bounding_boxes(self, image, suspect_data, image_original, suspect_img_encoded, suspect_feature1_encoded):
        original_size = [image_original.shape[1],image_original.shape[0]]
        resized_size = [image.shape[1],image.shape[0]]
        data = []
        Reid_flag = CameraWidget.frame_fetch

        for i,box in enumerate(suspect_data):
            box = self.scale_bbox(box,original_size,resized_size)
            data.append(box)

        self.encoded_img = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode() # Encoding for complete image..

        if Reid_flag is True: # need to turn this flag off
            encoded_frames_for_reid = CameraWidget.encoded_frames_for_reid
            s = time.time()
            json_response = self.session.post(self.url, json={'image_data': encoded_frames_for_reid,
                                            'suspect_img':  suspect_img_encoded,
                                            'suspect_feat_img': suspect_feature1_encoded})
            print('Time taken by server when reid is -True', time.time()-s)
            json_data = json_response.json()
            try:
                for key, value in json_data.items():
                    if key != 'ReidStatus':
                        camid = int(key)
                        self.camId = camid
                        self.new_bounding_box_cord = value
            except Exception as e:
                print('Exception reading camera and reid response...',e)

        else:
            if suspect_img_encoded is None:
                print('suspect_img_encoded is ', suspect_img_encoded)
            s = time.time()
            json_response = self.session.post(self.url, json={'image': self.encoded_img,
                                                        'suspect_img':  suspect_img_encoded,
                                                        'suspect_feat_img': suspect_feature1_encoded})
            print('Time taken by server when reid is -False', time.time()-s)
            try:
                json_data = json_response.json()
            except Exception as e:
                print('Exception json_response.json()', e)
                print('json response was..',json_response)
            try:
                self.do_ReID = json_data['ReidStatus']
            except Exception as e:
                pass

        if self.camId is not None and self.new_bounding_box_cord is not None:
            print('**********************Setting Reid to******************:', False)
            CameraWidget.frame_fetch = False # this will stop the extraction of frames from all the streams at client.
            status = 'new_position'
            data = [self.camId, [self.new_bounding_box_cord]]
            self.camId = None
            self.new_bounding_box_cord = None
            return status, data

        elif self.do_ReID is True:
            status = 'reid'
            self.do_ReID = False
            return status, None
        else:
            try:
                results = pd.DataFrame(json_data)
                results = results.T.values
                status = 'show_box'
                # print(results)
            except Exception as e:
                status = None
                results = None
                print('Exception incountered: ',e)

            # print(results.T.values)
            # status = 'show_box'
            return status, results


class CameraWidget(QWidget):
    frame_fetch = False
    encoded_frames_for_reid = {}
    active_cam = None
    suspect_img_encoded = None
    suspect_feature1_encoded = None

    def __init__(self, width, height, stream_link=0, id = None, aspect_ratio=False, parent=None, deque_size=1):
        super(CameraWidget, self).__init__(parent)
        # Initialize deque used to store frames read from the stream
        self.suspect_data = []

        self.suspect_img_encoded = None
        self.suspect_feature1_encoded = None
        self.deque = deque(maxlen=deque_size)
        self.arr = []
        self.frame = None
        self.url = 'http://127.0.0.1:5000/predict'
        self.yolo = YOLO()
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio
        self.cv_frame = None
        self.frame_id = id
        self.start_detection = False

        self.camera_stream_link = stream_link # path/URL

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QtWidgets.QLabel() # making object to display frame
        self.load_network_stream() # this will fire capture stream thread

        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=()) # this will read frame from stream in parallel
        self.get_frame_thread.daemon = True # running thread in background
        self.get_frame_thread.start()

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        # self.timer.setInterval(200)
        self.timer.timeout.connect(self.set_frame)
        self.timer.start()

        print('Started camera: {}'.format(self.camera_stream_link))


    def load_network_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):

                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True

        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def verify_network_stream(self, link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""
        while True:
            try:
                if self.capture.isOpened() and self.online:
                    # Read next frame from stream and insert into deque
                    status, frame = self.capture.read()
                    if status:
                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    # Attempt to reconnect
                    print('attempting to reconnect', self.camera_stream_link)
                    self.load_network_stream()
                    self.spin(2)
                self.spin(.001)
            except AttributeError:
                pass

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""
        label_text = 'Suspect'

        if not self.online:
            self.spin(1)
            return

        if self.deque and self.online:
            # Grab latest frame
            frame = self.deque[-1]
            self.cv_frame = frame
            self.frame = frame
            frame_text = 'Cam: ' + str(self.frame_id)

            # Keep frame aspect ratio
            if self.maintain_aspect_ratio:
                self.frame = imutils.resize(self.frame, width=self.screen_width)
            # Force resize
            else:
                self.frame = cv2.resize(self.frame, (self.screen_width, self.screen_height))

            if CameraWidget.frame_fetch is True:
                copied_frame = self.frame.copy()
                encoded_frame = base64.b64encode(cv2.imencode('.jpg', copied_frame)[1]).decode()
                CameraWidget.encoded_frames_for_reid[self.frame_id] = encoded_frame


            if CameraWidget.active_cam == self.frame_id:
                # if length of encoded frames is geq than zero we wana do reid...send all frames, suspect data for inference..
                # print(len(CameraWidget.encoded_frames_for_reid))
                status, inference_results = self.yolo.get_bounding_boxes(self.frame, self.suspect_data, frame, CameraWidget.suspect_img_encoded, CameraWidget.suspect_feature1_encoded)

                if status == 'reid': # re-identification block trigger
                    CameraWidget.frame_fetch = True

                elif status == 'new_position':
                    CameraWidget.active_cam = inference_results[0] # assigning new cam id for active camera.

                elif status == None:
                    pass

                elif status == 'show_box': # if inference has data then only draw on the select feed....
                    for df in inference_results:
                        x1,y1,x2,y2 = df[0],df[1],df[2],df[3]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        self.frame = cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 252, 124), 2)

                        text_width, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_width = text_width[0]
                        shaded_area_width = text_width + 10

                        if shaded_area_width > (x2 - x1):
                            font_scale = (x2 - x1) / shaded_area_width
                        else:
                            font_scale = 1
                        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

                        text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2
                        text_y = y1 - 10
                        self.frame=cv2.rectangle(self.frame, (text_x - 5, text_y - text_size[1] // 2 - 5), (text_x + text_size[0] + 5, text_y + text_size[1] // 2 + 5), (0, 255, 0), -1)

                        self.frame =cv2.putText(self.frame, label_text, (text_x, text_y + text_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


            # Add timestamp to cameras
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

            cv2.rectangle(self.frame, (self.screen_width - 150, 0), (self.screen_width, 50), color=(0, 0, 0),
                          thickness=-1)

            cv2.putText(self.frame, frame_text, (self.screen_width - 135, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)

            cv2.putText(self.frame, dt_string, (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)

            # Convert to pixmap and set to video frame
            self.img = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
            self.pix = QtGui.QPixmap.fromImage(self.img)
            self.video_frame.setPixmap(self.pix)

            # CameraWidget.encoded_frames_for_reid = {} #flushing frames..

    def get_video_frame(self):
        return self.video_frame, self.cv_frame

    def showImage(self, stream_id):
        if True:
            cv2.namedWindow('Select-ROI',0)
            # cv2.moveWindow('Select-ROI', 1100, 250)
            #cv2.resizeWindow('Select-ROI', 768, 576)
            frame = self.frame.copy()
            ROI_coords = cv2.selectROIs('Select-ROI', frame)

            for index, roi in enumerate(ROI_coords):
                if index == 0:
                    x1,y1,x2,y2 = roi
                    x2 = x1 + x2
                    y2 = y1 + y2
                    suspect_img = frame[y1:y2, x1:x2]
                    CameraWidget.suspect_img_encoded = base64.b64encode(cv2.imencode('.jpg', suspect_img)[1]).decode()
                    self.suspect_data.append(list(roi))
                else:
                    x1,y1,x2,y2 = roi
                    x2 = x1 + x2
                    y2 = y1 + y2
                    img = frame[y1:y2, x1:x2]
                    self.suspect_data.append(list(roi))
                    CameraWidget.suspect_feature1_encoded = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

            CameraWidget.active_cam = stream_id
            cv2.destroyAllWindows()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camWidget_list = []
        self.stream_id = None
        self.setWindowTitle("IVP-System")
        self.cw = QtWidgets.QWidget()
        self.ml = QtWidgets.QGridLayout()
        self.cw.setLayout(self.ml)
        self.setCentralWidget(self.cw)
        self.video_frame = QtWidgets.QLabel(self)
        self.video_frame.setGeometry(QtCore.QRect(0, 0, 600, 400))
        self.video_frame.setScaledContents(True)
        self.video_frame.setObjectName("video_frame")
        self.screen_width = QtWidgets.QApplication.desktop().screenGeometry().width()
        self.screen_height = QtWidgets.QApplication.desktop().screenGeometry().height()

        self.toolbar = QtWidgets.QToolBar("Toolbar")
        self.toolbar.setIconSize(QtCore.QSize(200, 100))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        self.input_field = QLineEdit(self)
        self.toolbar.addWidget(self.input_field)
        self.tracking_button = QPushButton('ROI-Selection', self)
        self.tracking_button.clicked.connect(self.selectROI)
        self.toolbar.addWidget(self.tracking_button)

    def selectROI(self):
        streamId = self.input_field.text()
        if streamId.isdigit():
            self.stream_id = int(streamId)
            for idx, cam in enumerate(self.camWidget_list):
                if idx == self.stream_id:
                    cam.showImage(self.stream_id)

    def setObjects(self,cam_widgets):
        for obj in cam_widgets:
            self.camWidget_list.append(obj)
        print('Setting camera widgets...!')



if __name__ == '__main__':
    flag = False
    app = QApplication([])

    # Create a Qt widget, which will be our window.
    window = MainWindow()
    screen_width = window.screen_width
    screen_height = window.screen_height
    # camera0 = '/home/vipin/projects/video_dataset/src_2.mp4'
    # camera1 = '/home/vipin/projects/video_dataset/src_3.mp4'
    # camera2 = '/home/vipin/projects/video_dataset/src_1.mp4'
    # camera3 = '/home/vipin/projects/video_dataset/src_0.mp4'
    # camera5 = '/home/vipin/projects/video_dataset/src_3.mp4'
    # camera4 = '/home/vipin/projects/video_dataset/src_2.mp4'
    # camera6 = '/home/vipin/projects/video_dataset/src_1.mp4'
    # camera7 = '/home/vipin/projects/video_dataset/src_0.mp4'
    # camera8 = '/home/vipin/projects/video_dataset/src_1.mp4'
    # camera9 = '/home/vipin/projects/video_dataset/originals/src_2.mp4'

    camera0 = '/home/vipin/Downloads/MTA_ext_short/test/cam_0/cam_0.mp4'
    camera1 = '/home/vipin/Downloads/MTA_ext_short/test/cam_1/cam_1.mp4'
    camera2 = '/home/vipin/Downloads/MTA_ext_short/test/cam_2/cam_2.mp4'
    camera3 = '/home/vipin/Downloads/MTA_ext_short/test/cam_3/cam_3.mp4'
    camera4 = '/home/vipin/Downloads/MTA_ext_short/test/cam_4/cam_4.mp4'
    camera5 = '/home/vipin/Downloads/MTA_ext_short/test/cam_5/cam_5.mp4'
    camera6 = '/home/vipin/Downloads/MTA_ext_short/test/cam_0/cam_0.mp4'
    camera7 = '/home/vipin/Downloads/MTA_ext_short/test/cam_1/cam_1.mp4'
    camera8 = '/home/vipin/Downloads/MTA_ext_short/test/cam_2/cam_2.mp4'
    # # camera9 = '/home/vipin/projects/video_dataset/originals/src_2.mp4'


    zero = CameraWidget(screen_width // 3, screen_height // 3, camera0,id=0)
    one = CameraWidget(screen_width // 3, screen_height // 3, camera1,id=1)
    two = CameraWidget(screen_width // 3, screen_height // 3, camera2,id=2)
    three = CameraWidget(screen_width // 3, screen_height // 3, camera3,id=3)
    four = CameraWidget(screen_width // 3, screen_height // 3, camera4,id=4)
    five = CameraWidget(screen_width // 3, screen_height // 3, camera5,id=5)
    # six = CameraWidget(screen_width // 4, screen_height // 4, camera6,id=6)
    # seven = CameraWidget(screen_width // 4, screen_height // 4, camera7,id=7)
    # eight = CameraWidget(screen_width // 4, screen_height // 4, camera8,id=8)
    # nine = CameraWidget(screen_width // 3, screen_height // 3, camera9,id=9)


    #cams = [zero, one, two, three]
    cams = [zero,one,two, three,four, five]
    window.setObjects(cams)
    window.ml.addWidget(zero.get_video_frame()[0], 0, 0, 1, 1)
    window.ml.addWidget(one.get_video_frame()[0], 0, 1, 1, 1)
    window.ml.addWidget(two.get_video_frame()[0], 0, 2, 1, 1)
    window.ml.addWidget(three.get_video_frame()[0], 1, 0, 1, 1)
    window.ml.addWidget(four.get_video_frame()[0], 1, 1, 1, 1)
    window.ml.addWidget(five.get_video_frame()[0], 1, 2, 1, 1)
    # window.ml.addWidget(six.get_video_frame()[0], 2, 0, 1, 1)
    # window.ml.addWidget(seven.get_video_frame()[0], 2, 1, 1, 1)
    # window.ml.addWidget(eight.get_video_frame()[0], 2, 2, 1, 1)
    # window.ml.addWidget(nine.get_video_frame()[0], 3, 1, 1, 1)

    # Start the event loop.
    window.show()
    app.exec_()
