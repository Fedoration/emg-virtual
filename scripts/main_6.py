# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:41:16 2024

@author: Юлия
"""

# %%
import sys
import numpy as np
import time
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
import threading
import queue
from continuous_prediction_6 import stream_processing  # Ensure this is the correct import path
from pylsl import StreamOutlet, StreamInfo


sys.path.append('C:\\Users\\bioel')
import utils.quats_and_angles as qa










channel_count = 64
chNames = [str(i) for i in range(1)]

class BCIStreamOutlet(StreamOutlet):
    def __init__(self, name = 'predictIrregular',srate = 500):
        info =  StreamInfo(name = name, type = 'IrregularQuats', channel_count= channel_count, nominal_srate = srate,
                           channel_format = 'float32', source_id = 'myuid34234')
        
        
        chns = info.desc().append_child('channels')
        for chname in chNames:
            ch = chns.append_child("channel")
            ch.append_child_value("label",chname)
            
            super(BCIStreamOutlet, self).__init__(info,chunk_size = 1)


class ImageApp(QWidget):
    def __init__(self):
        
        
        super().__init__()
        
        
        self.outlet = BCIStreamOutlet()
        
        self.initUI()
        self.start_time = time.time()
        self.last_time = time.time()
        self.dct = []
        #self.PATH = 'D:/Education/ВКР/Data/2024-05-10 (final-online)/'

        # Initialize the queue and start the stream processing thread
        self.queue_output = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=stream_processing,
            args=(self.queue_output, self.stop_event)
        )
        
        
        self.thread.start()
        # Set up a timer to periodically check the queue
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(5)  # Check every 5 ms
        
    def initUI(self):
        self.setWindowTitle('REAL TIME')
        self.setGeometry(0, 40, 1000, 1000)
        self.setStyleSheet("background-color: black;")  # Set background color to black


    '''
        # Load and set up the images
        self.cloud_label = QLabel(self)
        self.cloud_label.setPixmap(QPixmap('resource/cloud.png'))
        self.cloud_label.setGeometry(10, 410, 250, 250)
        self.cloud_label.setScaledContents(True)
        self.cloud_label.setStyleSheet("background-color: transparent;")

        self.clock_label = QLabel(self)
        self.clock_label.setPixmap(QPixmap('resource/clock.png'))
        self.clock_label.setGeometry(360, 10, 250, 250)
        self.clock_label.setScaledContents(True)
        self.clock_label.setStyleSheet("background-color: transparent;")

        self.calendar_label = QLabel(self)
        self.calendar_label.setPixmap(QPixmap('resource/calendar.png'))
        self.calendar_label.setGeometry(410, 700, 150, 150)
        self.calendar_label.setScaledContents(True)
        self.calendar_label.setStyleSheet("background-color: transparent;")

        self.heart_label = QLabel(self)
        self.heart_label.setPixmap(QPixmap('resource/heart.png'))
        self.heart_label.setGeometry(750, 400, 200, 200)
        self.heart_label.setScaledContents(True)
        self.heart_label.setStyleSheet("background-color: transparent;")

        # Label to show the selected image
        self.selected_label = QLabel(self)
        self.selected_label.setGeometry(350, 320, 300, 300)
        self.selected_label.setScaledContents(True)
        self.selected_label.setStyleSheet("background-color: transparent;")

        # List of images and their labels
        self.images = [self.cloud_label, self.clock_label, self.calendar_label, self.heart_label]
        self.current_index = 0  # Index of the currently selected image

        # Update the selected label to show the initial selected image
        self.update_selected_label()

        self.show()
    '''
    def closeEvent(self, event):
        self.stop_event.set()
        self.thread.join()
        event.accept()

    def process_queue(self):
        if self.queue_output.empty():
            return

        key = self.queue_output.get()
        
        #self.reset_image_positions()
        
        print(key)
        
        '''
        if key == 3:  # Right
            print('extendthumb')
            self.selected_label.setPixmap(self.heart_label.pixmap())
        elif key == 2:  # Left
            print('extendthree')
            self.selected_label.setPixmap(self.cloud_label.pixmap())
        elif key == 4:  # Up
            print('flexfirst')
            self.selected_label.setPixmap(self.clock_label.pixmap())
        elif key == 0:  
            print('extenrfirst')
            self.zoomIn()
        elif key == 1:  # Down
            print('extendfist')
            self.selected_label.setPixmap(self.calendar_label.pixmap())
        elif key == 5:  # Left
            print('flexthree')
            self.selected_label.setPixmap(self.cloud_label.pixmap())
        elif key == 6:  # Down
            print('flexthumb')
            self.selected_label.setPixmap(self.calendar_label.pixmap())
        elif key == 7:  # Left
            print('makefist')
            self.selected_label.setPixmap(self.cloud_label.pixmap())
        else:
            # Noise or unrecognized command
            pass'''
        
        
        # TRUE DATA!!!!!!!!!!!!!
        #self.outlet.push_chunk(key)
        # FAKE DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.outlet.push_chunk(np.random.rand(64).tolist())
        
        
        angles = np.array(key)[None,:]
        
        
        
        
        
        
        
        
        #angles = (np.random.rand(20)[None,:]-0.5)*2*np.pi
        quats = qa.get_quats(angles)
        
        quats_64 = np.reshape(quats, (quats.shape[0], 64))
        
        
        self.outlet.push_chunk(quats_64.tolist())
        
 #['extenrfirst', 'extendfist', 'extendthree', 'extendthumb', 'flexfirst', 'flexthree', 'flexthumb', 'makefist']
                         

        # Log the action
        #self.write_data(key)
    '''
    def move_selection(self, delta):
        # Update the current index and wrap around if necessary
        self.current_index = (self.current_index + delta) % len(self.images)
        self.update_selected_label()

    def update_selected_label(self):
        # Update the selected_label to show the currently selected image
        selected_image_label = self.images[self.current_index]
        self.selected_label.setPixmap(selected_image_label.pixmap())
    
    def zoomIn(self):
        pixmap = self.selected_label.pixmap()
        if pixmap:
            img = pixmap.toImage()
            painter = QPainter(img)
            pen = QPen(Qt.white)
            pen.setWidth(8)
            painter.setPen(pen)
            painter.drawEllipse(img.rect())
            painter.end()
            self.selected_label.setPixmap(QPixmap.fromImage(img))
        
    def write_data(self, key):
        actual_time = time.time()
        inter_time = actual_time - self.last_time  # Time between events
        act_time = actual_time - self.start_time  # Time after start
        d = {'key': key, 'inter_time': inter_time, 'act_time': act_time}
        self.dct.append(d)
        self.last_time = actual_time

    def csv_writer(self, dct):
        keys = ['key', 'inter_time', 'act_time', 'whole_time']
        whole_path = self.PATH + str(self.start_time) + '.csv'
        with open(whole_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dct)'''
    '''
    def reset_image_positions(self):
        # Reset all images to their original positions
        self.cloud_label.setGeometry(10, 410, 250, 250)  # Reset to original pos
        # Reset other images if needed

    def move_cloud_to_center(self):
        # Move cloud image to the center of the window
        center_x = (self.width() - self.cloud_label.width()) // 2
        center_y = (self.height() - self.cloud_label.height()) // 2
        self.cloud_label.setGeometry(center_x, center_y, 250, 250)
    '''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageApp()
    sys.exit(app.exec_())