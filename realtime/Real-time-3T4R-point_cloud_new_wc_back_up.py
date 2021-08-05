from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from offline_process_3t4r_for_correct import DataProcessor_offline
from tkinter import filedialog
import socket
import tkinter as tk
from queue import Queue
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import numpy as np
import threading
import sys
import read_binfile
import time
import cv2
from camera_offine import CamCapture
# -----------------------------------------------
from real_time_pd_qthread import UdpListener, DataProcessor
from radar_config import SerialConfig
from ultis import send_cmd,get_color,ConnectDca
from Webcam_Save_Vedio_Qthread import RTSPVideoWriterObject
# -----------------------------------------------
from pd_layout_new_wbcam import Ui_MainWindow

set_radar = SerialConfig(name='ConnectRadar', CLIPort='COM13', BaudRate=115200)
config = '../radar_config/xwr68xx_profile_2021_03_23T08_12_36_405.cfg'

class Realtime_sys():

    def __init__(self):
        # Host setting
        address = ('192.168.33.30', 4098)
        buff_size = 2097152
        adc_sample = 64
        chirp = 16
        tx_num = 3
        rx_num = 4
        self.radar_config = [adc_sample, chirp, tx_num, rx_num]
        frame_length = adc_sample * chirp * tx_num * rx_num * 2
        self.save_frame_len = 1000
        self.pd_save = []
        self.pd = []
        self.rdi = []
        self.rai = []
        self.raw = []
        # ------------- call init class -------------
        self.bindata = Queue()
        self.rawdata = Queue()
        self.collector = UdpListener('Listener', frame_length, address, buff_size,self.bindata,self.rawdata)
        self.processor = DataProcessor('Processor', self.radar_config, self.bindata, "0105", status=0 )
        self.processor.data_signal.connect(self.Qthreadupdate_fig)
        self.pd_save_status = 0
        # ------------- mode setting  -------------
        self.camera = True
        self.btn_status = False
        self.run_state  = False
        self.sure_next = True
        self.sure_image = True
        self.frame_count = 0
        self.data_proecsss = DataProcessor_offline()
        self.path = 'C:/Users/user/Desktop/thmouse_training_data/'\
        #----- for test Usage ----
        self.sure_select = False
        self.realtime_mdoe = True
        self.rai_mode =  0

    def restart(self):
        self.cam1_thread.restart_videowriter()
        self.cam2_thread.restart_videowriter()

    def restart_radar(self):
        # self.collector.quit()
        # self.collector.data_socket.close()
        set_radar.StopRadar()
        # self.ConnectDca1000()
        # adc_sample = 64
        # chirp = 16
        # tx_num = 3
        # rx_num = 4
        # frame_length = adc_sample * chirp * tx_num * rx_num * 2
        # # Host setting
        # address = ('192.168.33.30', 4098)
        # buff_size = 2097152
        # self.save_frame_len = 200
        # self.collector = UdpListener('Listener', frame_length, address, buff_size,self.bindata,self.rawdata)
        # self.collector.start()


    def append_rawdata(self,rawdata):
        self.raw.append(rawdata)
        # print(np.shape(self.raw))

    def Qthreadupdate_fig(self,rdi,rai,pd):
        self.processor.Sure_staic_RM = self.static_rm.isChecked()
        # self.processor.Sure_larry_staic_RM = self.larry_static_rm.isChecked()
        self.pd = pd
        if self.pd_save_status == 1  :
            self.raw.append(self.rawdata.get())
            # self.frame_count_lb.setText(str(self.frame_count))
            self.frame_count += 1
            if self.frame_count>=self.save_frame_len:
                self.StopRecord()
                self.SaveData()
        # if not rai.empty():
        self.img_rdi.setImage(np.rot90(rdi, 1))
        # if not rai.empty():
        self.img_rai.setImage(np.fliplr(np.flip(rai, axis=0)).T)
        # if not pd.empty():
        pos = np.transpose(pd, [1, 0])
        # print(pos)
        p13d.setData(pos=pos[:, :3], color=[1, 0.35, 0.02, 1], pxMode=True)

    def save_process(self):
        self.raw = np.append(self.raw, self.rawData.get())
        self.frame_count += 1
        if self.frame_count >= self.save_frame_len:
            self.StopRecord()


    def update_cam1(self,img):
        self.image_label1.setPixmap(img)

    def update_cam2(self,img):
        self.image_label2.setPixmap(img)

    def start(self):
        self.run_state = True
        self.frame_count = 0
        self.stop_btn.setText("stop")
        if self.sure_image:
            self.th_cam1 = CamCapture(self.file_path_cam1)
            self.th_cam2 = CamCapture(self.file_path_cam2)
        self.update_figure()

    def stop(self):
        if self.run_state:
            self.stop_btn.setText("Continues")
        else:
            self.stop_btn.setText("stop")
        self.run_state = not(self.run_state)
        self.sure_next = True
        self.update_figure()

    def RDI_update(self):
        if self.realtime_mdoe:
            # rd = self.RDIData.get()
            if not RDIData.empty():
                rd = RDIData.get()
                if self.Sure_staic_RM == False:
                    self.img_rdi.setImage(np.rot90(rd, 1))
                else:
                    self.img_rdi.setImage(np.rot90(rd, 1), levels=[40, 150])

    def RAI_update(self):
        global count, view_rai, p13d

        if self.realtime_mdoe:
            if not RAIData.empty():
                a = RAIData.get()
                if self.Sure_staic_RM == False:
                    self.img_rai.setImage(np.fliplr(np.flip(a, axis=0)).T)
                    # ------no static remove------
                    # self.img_rai.setImage(np.fliplr(np.flip(a, axis=0)).T, levels=[15e4, 50.0e4])
                else:
                    # ------ static remove ------
                    self.img_rai.setImage(np.fliplr(np.flip(a, axis=0)).T, levels=[0.5e2, 9.0e4])

    def PD_update(self):
        global count, view_rai, p13d,nice,ax
        # --------------- plot 3d ---------------
        if self.realtime_mdoe:
            # if not pointcloud.empty():
            pos = pointcloud.get()
            pos = np.transpose(pos, [1, 0])
            p13d.setData(pos=pos[:, :3], color=[1, 0.35, 0.02, 1], pxMode=True)
            if self.pd_save_status == 1:
                self.save_process()

    def update_figure(self):
        global count,view_rai,p13d
        self.Sure_staic_RM = self.static_rm.isChecked()

        if self.realtime_mdoe:
            QtCore.QTimer.singleShot(1, self.RDI_update)
            QtCore.QTimer.singleShot(1, self.RAI_update)
            QtCore.QTimer.singleShot(1, self.PD_update)
            QtCore.QTimer.singleShot(1, self.update_figure)
            QApplication.processEvents()
        else:
            if self.run_state:
                self.RDI ,self.RAI,self.RAI_ele,self.PD = self.data_proecsss.run_proecss(self.rawData[self.frame_count],\
                                                                self.rai_mode,self.Sure_staic_RM,self.chirp)
                self.RDI_update()
                self.RAI_update()
                self.PD_update()
                time.sleep(0.05)
                if self.sure_next:
                    self.frame_count +=1
                    QtCore.QTimer.singleShot(1, self.update_figure)
                    QApplication.processEvents()
                if self.sure_image:
                    self.image_label1.setPixmap((self.th_cam1.get_frame()))
                    self.image_label2.setPixmap((self.th_cam2.get_frame()))
            else :
                pass

    def pre_frame(self):
        if self.frame_count >0:
            self.frame_count -=1
            self.sure_next = False
            self.run_state=True
            self.update_figure()

    def next_frame(self):
        if self.frame_count<=self.frame_total_len:
            self.frame_count += 1
            self.sure_next = False
            self.run_state = True
            self.update_figure()

    def SelectFolder(self):
        root = tk.Tk()
        root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:/kaiku_report/2021-0418for_posheng/')
        if self.sure_select == True:
            self.file_path = filedialog.askopenfilename(parent=root, initialdir=self.path)
            self.browse_text.setText(self.file_path)
        else:
            self.file_path = self.path + 'raw.npy'
            self.browse_text.setText(self.file_path)

        return self.file_path

    def SelectFolder_cam1(self):
        root = tk.Tk()
        root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        if self.sure_select == True:
            self.file_path_cam1 = filedialog.askopenfilename(parent=root, initialdir=self.path)
            self.browse_text_cam1.setText(self.file_path_cam1)
        else:
            self.file_path_cam1 = self.path + 'output0.mp4'
            self.browse_text_cam1.setText(self.file_path_cam1)

        return self.file_path_cam1

    def SelectFolder_cam2(self):
        root = tk.Tk()
        root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        if self.sure_select == True:
            self.file_path_cam2 = filedialog.askopenfilename(parent=root, initialdir=self.path)
            self.browse_text_cam2.setText(self.file_path_cam2)
        else:
            self.file_path_cam2 = self.path + 'output1.mp4'
            self.browse_text_cam2.setText(self.file_path_cam2)
        return self.file_path_cam2

    def enable_btns(self,state):
        self.pre_btn.setEnabled(state)
        self.next_btn.setEnabled(state)
        self.start_btn.setEnabled(state)
        self.stop_btn.setEnabled(state)

    def slot(self, object):
        print("Key was pressed, id is:", self.radio_group.id(object))
        '''
        raimode /0/1/2:
                0 -> FFT-RAI
                1 -> beamformaing RAI 
                3 -> static clutter removal
        '''
        self.rai_mode = self.radio_group.id(object)

        if self.rai_mode ==1:
            self.view_rai.setRange(QtCore.QRectF(10, 0, 170, 80))
        else:
            self.view_rai.setRange(QtCore.QRectF(-5, 0, 100, 60))

    def load_file(self):
        load_mode = 1
        if load_mode == 0 :
            self.rawData =read_binfile.read_bin_file(self.file_path,[64,64,32,3,4],mode=0,header=False,packet_num=4322)
            self.rawData = np.transpose(self.rawData,[0,1,3,2])
            self.chirp = 32
        elif load_mode == 1:
            data  =np.load(self.file_path,allow_pickle=True)
            data = np.reshape(data, [-1, 4])
            data = data[:, 0:2:] + 1j * data[:, 2::]
            self.rawData = np.reshape(data,[-1,48,4,64])
            # print(len(self.rawData))
            self.frame_total_len = len(self.rawData)
            self.chirp = 16

        self.enable_btns(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
           print("fuck u")

    def StartRecord(self):
        self.starttime = time.time()
        self.processor.status = 1
        self.collector.status = 1
        if self.camera == True:
            self.cam1_thread.record = True
            self.cam2_thread.record = True
        self.pd_save_status = 1
        print('Start Record Time:', (time.ctime(time.time())))
        print('=======================================')

    def StopRecord(self):
        total= time.time()- self.starttime
        fps = self.save_frame_len/total
        print("FPS is {}".format(fps))
        self.processor.status = 0
        self.collector.status = 0
        self.pd_save_status = 0
        if self.camera == True:
            self.cam1_thread.record = False
            self.cam2_thread.record = False
            self.cam1_thread.release_video()
            self.cam2_thread.release_video()
        print('Stop Record Time:', (time.ctime(time.time())))
        print('=======================================')

    def ConnectDca1000(self):
        # dca1000  = ConnectDca()
        # dca1000.start()
        global sockConfig, FPGA_address_cfg
        print('Connect to DCA1000')
        print('=======================================')
        config_address = ('192.168.33.30', 4096)
        FPGA_address_cfg = ('192.168.33.180', 4096)
        cmd_order = ['9', 'E', '3', 'B', '5', '6']
        sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockConfig.bind(config_address)
        for k in range(5):
            # Send the command
            sockConfig.sendto(send_cmd(cmd_order[k]), FPGA_address_cfg)
            time.sleep(0.1)
            # Request data back on the config port
            msg, server = sockConfig.recvfrom(2048)
            # print('receive command:', msg.hex())
        sockConfig.close()

    def openradar(self):
        set_radar.StopRadar()
        set_radar.SendConfig(config)
        self.collector.start()
        self.processor.start()

        # self.update_figure()
        print('=============openradar=================')


    def exit(self):
        if self.camera == True:
            self.cam1_thread.quit()
            self.cam2_thread.quit()
        set_radar.StopRadar()

        self.app.instance().exit()

    def SaveData(self):
        print("save the npy")
        np.save("C:/Users/user/Desktop/thumouse_training_data_new/raw.npy", self.raw)
        self.raw = []
        self.frame_count = 0

    def calulate1(self,arr): # cam1--> left side camera x1{none_use},y1{z}
        scale_x1 = 56 / 640 * 0.015  # cm / pixel * (0.015 point/cm)
        scale_y1 = 44 / 480 * 0.015
        x1 = np.array(arr[0])
        y1 = np.array(arr[1])
        if (y1 != None).all() == True:
            y1 = y1.astype(np.double)
            y1 = np.round((y1 * scale_y1), 3)
            y1 -= ((480 * scale_y1) / 2)
            if self.pd != [] :
                tmp = np.mean(self.pd, axis=1)
                dif = (tmp[2]-y1[8]*-1)
                print("\n ================")
                print("校正z:{}  radarZ:{}  cam:{}".format(dif/0.015,tmp[2]/0.015, y1[8]*-1/0.015))

    def calulate2(self,arr): # cam1--> above camera x2{x},y2{y}
        scale_x2 = 45 / 640 * 0.015
        scale_y2 = 33 / 480 * 0.015
        x2 = np.array(arr[0])
        y2 = np.array(arr[1])
        x2 = np.where(x2 is not None, x2, np.double(320))
        x2 = x2.astype(np.double)
        x2 = (x2 * scale_x2)
        x2 -= ((640 * scale_x2) / 2)
        x2 = np.round(x2, 3)
        x2 += 0.015

        y2 = np.where(y2 is not None, y2, np.double(480))
        y2 = y2.astype(np.double)
        y2 = (y2 * scale_y2)
        y2 = (y2*-1) + ((480 * scale_y2)) + 0.015*8
        y2 = np.round(y2, 3)
        # y2 = y2 + 0.30

        if self.pd != [] :
            tmp = np.mean(self.pd, axis=1)
            difx = (tmp[0]-x2[8]*-1)
            dify = (tmp[1]-y2[8])
            yy= np.min(self.pd[1,:])
            # print("\n ================")
            print("校正x:{}  radarx:{}  camx:{}".format(difx/0.015  ,tmp[0], x2[8]*-1/0.015))
            print("\n校正y:{}  radary:{}  camy:{}".format(dify/0.015,yy, y2[8]/0.015))

    def open_camera(self):
        self.save_frame_len  = int(self.camera_frame_count_lb.toPlainText())
        self.cam1_thread = RTSPVideoWriterObject(0, "vedio1", save_frame_len=self.save_frame_len, mediapipe_mode=0)
        self.cam2_thread = RTSPVideoWriterObject(1, "vedio2", save_frame_len=self.save_frame_len, mediapipe_mode=0)
        self.cam1_thread.change_pixmap_signal.connect(self.update_cam1)  # Qthread link slot
        self.cam2_thread.change_pixmap_signal.connect(self.update_cam2)
        self.cam1_thread.hand_points.connect(self.calulate1)  # Qthread link slot
        self.cam2_thread.hand_points.connect(self.calulate2)  # Qthread link slot
        self.cam1_thread.start()
        self.cam2_thread.start()
        self.camera = True

    def build_GLline(self,p1,p2):
        x1 = p1[0];y1 = p1[1];z1 = p1[2]
        x2 = p2[0];y2 = p2[1];z2 = p2[2]
        return  gl.GLLinePlotItem(pos=np.array([[[x1, y1, z1], [x2, y2, z2]]]), color=[128, 255, 128, 255], antialias=False)

    def bounding_box(self,view_PD):
        # half_len = 0.1875
        # all_len = 0.1875*2
        all_len = 0.234375
        half_len = all_len/2
        self.line1 = self.build_GLline([half_len,       0,    half_len], [-1*half_len,       0,    half_len])
        self.line2 = self.build_GLline([half_len, all_len,    half_len], [-1*half_len, all_len,    half_len])
        self.line3 = self.build_GLline([half_len,       0, -1*half_len], [-1*half_len,       0, -1*half_len])
        self.line4 = self.build_GLline([half_len, all_len, -1*half_len], [-1*half_len, all_len, -1*half_len])

        self.line5 = self.build_GLline([half_len,       0,    half_len], [half_len,       0, -1*half_len])
        self.line6 = self.build_GLline([half_len,       0,    half_len], [half_len, all_len,    half_len])
        self.line7 = self.build_GLline([half_len, all_len,    half_len], [half_len, all_len, -1*half_len])
        self.line8 = self.build_GLline([half_len,       0, -1*half_len], [half_len, all_len, -1*half_len])

        self.line9  = self.build_GLline([-1*half_len,       0,    half_len], [-1*half_len,       0, -1*half_len])
        self.line10 = self.build_GLline([-1*half_len,      0,    half_len], [-1*half_len, all_len,    half_len])
        self.line11 = self.build_GLline([-1*half_len, all_len,    half_len], [-1*half_len, all_len, -1*half_len])
        self.line12 = self.build_GLline([-1*half_len,      0, -1*half_len], [-1*half_len, all_len, -1*half_len])

        view_PD.addItem(self.line1)
        view_PD.addItem(self.line2)
        view_PD.addItem(self.line3)
        view_PD.addItem(self.line4)
        view_PD.addItem(self.line5)
        view_PD.addItem(self.line6)
        view_PD.addItem(self.line7)
        view_PD.addItem(self.line8)
        view_PD.addItem(self.line9)
        view_PD.addItem(self.line10)
        view_PD.addItem(self.line11)
        view_PD.addItem(self.line12)

    def plot(self):
        global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv, img_cam, savefilename,view_rai,p13d,nice
        # ---------------------------------------------------
        self.app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        MainWindow.show()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        self.browse_btn = ui.browse_btn
        self.browse_text = ui.textEdit
        self.browse_text_cam1 = ui.textEdit_cam1
        self.browse_text_cam2 = ui.textEdit_cam2
        self.load_btn = ui.load_btn
        self.start_btn = ui.start_btn
        self.stop_btn  =ui.stop_btn
        self.next_btn  =ui.next_btn
        self.pre_btn = ui.pre_btn
        self.radio_group =  ui.radio_btn_group
        self.static_rm = ui.sure_static
        # self.larry_static_rm = ui.sure_larry
        # self.larry_static_rm.setEnabled(False)
        self.cam1 = ui.label_cam1
        self.cam2 = ui.label_cam2
        self.cam1_btn =  ui.browse_cam1_btn
        self.cam2_btn =  ui.browse_cam2_btn
        self.image_label1 =ui.image_label1
        self.image_label2 =ui.image_label2
        #----------------- realtime btn clicked connet -----------------
        self.start_dca_rtbtn = ui.dca1000_rtbtn
        self.send_cfg_rtbtn = ui.sendcfg_rtbtn
        self.record_rtbtn = ui.record_rtbtn
        self.stop_rtbtn = ui.stop_rtbtn
        self.save_rtbtn = ui.save_rtbtn
        self.exit_rtbtn = ui.exit_rtbtn
        self.restart_rtbtn = ui.restart_rtbtn
        self.restart_radar_rtbtn = ui.restart_radar_rtbtn
        self.set_camera_true_btn = ui.btn_frame_count
        self.camera_frame_count_lb = ui.edit_frame_count
        self.frame_count_lb = ui.label_frame_count1
        self.start_dca_rtbtn.clicked.connect(self.ConnectDca1000)
        self.send_cfg_rtbtn.clicked.connect(self.openradar)
        self.record_rtbtn.clicked.connect(self.StartRecord)
        self.exit_rtbtn.clicked.connect(self.exit)
        self.save_rtbtn.clicked.connect(self.SaveData)
        self.restart_rtbtn.clicked.connect(self.restart)
        self.restart_radar_rtbtn.clicked.connect(self.restart_radar)
        self.set_camera_true_btn.clicked.connect(self.open_camera)
        #----------------- btn clicked connet -----------------
        self.browse_btn.clicked.connect(self.SelectFolder)
        self.cam1_btn.clicked.connect(self.SelectFolder_cam1)
        self.cam2_btn.clicked.connect(self.SelectFolder_cam2)
        self.load_btn.clicked.connect(self.load_file)
        self.start_btn.clicked.connect(self.start)
        self.next_btn.clicked.connect(self.next_frame)
        self.pre_btn.clicked.connect(self.pre_frame)
        self.stop_btn.clicked.connect(self.stop)
        self.radio_group.buttonClicked.connect(self.slot)
        # # -----------------------------------------------------
        self.view_rdi = ui.graphicsView.addViewBox()
        self.view_rai = ui.graphicsView_2.addViewBox()
        view_PD = ui.graphicsView_3
        # ---------------------------------------------------
        # lock the aspect ratio so pixels are always square
        self.view_rdi.setAspectLocked(True)
        self.view_rai.setAspectLocked(True)
        self.img_rdi = pg.ImageItem(border='w')
        self.img_rai = pg.ImageItem(border='w')
        self.img_rai_ele = pg.ImageItem(border='w')
        img_cam = pg.ImageItem(border='w')
        #-----------------
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view_PD.addItem(xgrid)
        view_PD.addItem(ygrid)
        view_PD.addItem(zgrid)
        xgrid.translate(0,10,-10)
        ygrid.translate(0, 0, 0)
        zgrid.translate(0, 10, -10)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)

        p13d = gl.GLScatterPlotItem(pos = np.zeros([1,3]) ,color=[50, 50, 50, 255])
        origin = gl.GLScatterPlotItem(pos = np.zeros([1,3]),color=[255, 0, 0, 255])
        coord = gl.GLAxisItem(glOptions="opaque")
        coord.setSize(10, 10, 10)
        view_PD.addItem(p13d)
        view_PD.addItem(coord)
        view_PD.addItem(origin)

        view_PD.orbit(45,6)
        view_PD.pan(1,1,1,relative=1)
        # self.lineup(view_PD)
        self.bounding_box(view_PD)
        self.enable_btns(False)
        # ang_cuv = pg.PlotDataItem(tmp_data, pen='r')
        # Colormap
        position = np.arange(64)
        position = position / 64
        position[0] = 0
        position = np.flip(position)
        colors = get_color()
        colors = np.flip(colors, axis=0)
        color_map = pg.ColorMap(position, colors)
        lookup_table = color_map.getLookupTable(0.0, 1.0, 256)
        self.img_rdi.setLookupTable(lookup_table)
        self.img_rai.setLookupTable(lookup_table)
        self.img_rai_ele.setLookupTable(lookup_table)
        self.view_rdi.addItem(self.img_rdi)
        self.view_rai.addItem(self.img_rai)
        self.view_rai.addItem(self.img_rai_ele)
        self.view_rdi.setRange(QtCore.QRectF(0, 0, 30, 70))
        self.view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
        updateTime = ptime.time()

        self.app.instance().exec_()

    def lineup(self,view_PD):
        ###---------------------------------------------
        self.hand = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[0, 255, 0, 255], pxMode=True)
        view_PD.addItem(self.hand)
        self.indexfinger = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        view_PD.addItem(self.indexfinger)
        self.thumb = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        view_PD.addItem(self.thumb)
        origin = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0.075, 0], [0, 0.075 * 2, 0], [0, 0.075 * 3, 0], [0, 0.075 * 4, 0], [0, 0.075 * 5, 0],
             [0, 0.075 * 6, 0]]), color=[255, 255, 255, 255])
        origin1 = gl.GLScatterPlotItem(pos=np.array(
            [[0.075 * -3, 0.3, 0], [0.075 * -2, 0.3, 0], [0.075 * -1, 0.3, 0], [0.075 * 1, 0.3, 0],
             [0.075 * 2, 0.3, 0], [0.075 * 3, 0.3, 0]]), color=[255, 255, 255, 255])
        origin2 = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0.3, 0.075 * -3], [0, 0.3, 0.075 * -2], [0, 0.3, 0.075 * -1], [0, 0.3, 0.075 * 1],
             [0, 0.3, 0.075 * 2], [0, 0.3, 0.075 * 3]]), color=[255, 255, 255, 255])
        view_PD.addItem(origin)
        view_PD.addItem(origin1)
        view_PD.addItem(origin2)
        origin_P = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0, 0]]), color=[255, 0, 0, 255])
        view_PD.addItem(origin_P)
        self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075 * 10, 0]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.hand_liney = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075 * 10, 0]]]),
                                            color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_2 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075 * 2], [0.075 * 8, 0.075 * 4, 0.075 * 2]]]),
                                              color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_1 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075*1], [0.075 * 8, 0.075 * 4, 0.075*1]]]),
                                            color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_d1 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075 * -1], [0.075 * 8, 0.075 * 4, 0.075 * -1]]]),
                                            color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_d2 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075 * -2], [0.075 * 8, 0.075 * 4, 0.075 * -2]]]),
                                            color=[128, 255, 128, 255], antialias=False)

        self.hand_linex = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0], [0.075 * 8, 0.075 * 4, 0]]]),
                                            color=[128, 255, 128, 255], antialias=False)

        self.hand_linez = gl.GLLinePlotItem(pos=np.array([[[0, 0.075 * 4, -0.075 * 8], [0, 0.075 * 4, 0.075 * 8]]]),
                                            color=[0.5,0.5,0.9,1], antialias=False)
        view_PD.addItem(self.hand_line)
        view_PD.addItem(self.hand_liney)
        view_PD.addItem(self.hand_linez)
        view_PD.addItem(self.hand_linex)



if __name__ == '__main__':
    print('======Real Time Data Capture Tool======')
    count = 0
    realtime = Realtime_sys()
    plotIMAGE = threading.Thread(target=realtime.plot())
    plotIMAGE.start()
    print("Program Close")
    set_radar.StopRadar()
    sys.exit()