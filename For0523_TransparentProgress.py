# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:23:57 2023

@author: 23842
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 01:19:32 2023

@author: 23842
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:47:51 2023

@author: 23842
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:46:23 2023

@author: 23842
"""

#%% UI
import sys
import os

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QFileDialog, QDesktopWidget, QCheckBox
from PyQt5.QtWidgets import QFormLayout, QLineEdit, QLabel, QListWidget, QListWidgetItem, QTextEdit
from PyQt5.QtGui import QPixmap, QImage

from PyQt5 import QtWidgets
#%% Code

#import os
import cv2
from pytesseract import pytesseract

#以下import与ts ocr无关
import numpy as np
import math
from skimage import measure
import matplotlib.pyplot as plt
from PIL import Image
import time
from scipy import ndimage
from multiprocessing import Pool

from numba import jit

#%% UI
class MyUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle("MyUI")
        self.setFixedSize(1600, 1200) #SizeFixed
        
        # Get the size of the user's screen
        screen_size = QDesktopWidget().screenGeometry(-1)
        self.move(round((screen_size.width() - self.width()) / 2), round((screen_size.height() - self.height()) / 2) )

        # Create main widget and layout
        main_widget = QtWidgets.QWidget()
        main_widget_layout = QtWidgets.QHBoxLayout()
        main_widget.setLayout(main_widget_layout)

        # Create left-side widget and layout
        left_widget = QtWidgets.QWidget()
        left_widget_layout = QtWidgets.QVBoxLayout()
        left_widget.setLayout(left_widget_layout)
        
        # Create buttons
        button_1 = QtWidgets.QPushButton("Import Image")
        button_1_6 = QtWidgets.QPushButton("Select\nAddress")
        button_2 = QtWidgets.QPushButton("Extract")
        button_3 = QtWidgets.QPushButton("Export")
        button_3_5 = QtWidgets.QPushButton("Get Progress Image")
        button_4 = QtWidgets.QPushButton("Get Image\nfor Trainning")
        button_6 = QtWidgets.QPushButton("Reco Tune\n(Beta)")
        
        # Set button sizes
        button_1.setFixedSize(240,120)
        button_1_6.setFixedSize(100,80)
        button_2.setFixedSize(240,120)
        button_3.setFixedSize(240,120)
        button_3_5.setFixedSize(240,80)
        button_4.setFixedSize(240,80)
        button_6.setFixedSize(240,80)
        
        
        self.Fix_Output_Address_button = QCheckBox('Switch')
        self.Fix_Output_Address_button.setText('Lock')
        self.Fix_Output_Address_button.setFixedSize(100,50)
        
        # Apply custom styles to the switch button
        self.Fix_Output_Address_button.setStyleSheet('''
            QCheckBox {
                background-color: rgba(255, 255, 255, 0.5);
                color: rgba(0, 0, 0, 0.7);
                padding: 3px;
                border-radius: 10px;
                font-size: 22px;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
                border-radius: 12px;
            }
            QCheckBox::indicator:checked {
                background-color: rgba(45, 165, 145, 0.5);
            }
        ''')
        
        # Add buttons to left-side widget layout
        left_widget_layout.addWidget(button_1)
        left_widget_layout.addWidget(button_2)
        left_widget_layout.addWidget(button_3)
        left_widget_layout.addWidget(button_3_5)
        left_widget_layout.addWidget(button_4)
        left_widget_layout.addWidget(button_6)

        # Create right-side widget and layout
        self.right_widget = QListWidget(self)
        # Set the style sheet for the QListWidget
        self.right_widget.setFixedSize(600,1130)
        self.right_widget.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
        """)
         
        # Create mid widget and layout
        self.mid_widget = QtWidgets.QWidget()
        self.mid_widget_layout = QtWidgets.QGridLayout()
        self.mid_widget.setLayout(self.mid_widget_layout)
        
        # Create file name label and text box
        filename_label = QtWidgets.QLabel("File Name")
        self.filename_textbox = QtWidgets.QTextEdit()
        self.filename_textbox.setFixedSize(628, 48)
        self.filename_textbox.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-radius: 10px; font-size: 32px;")
        self.filename_textbox.setText("")
        
        # Create output address label and text box
        output_address_label = QtWidgets.QLabel("Output Address")
        self.OutputAddress_textbox = QtWidgets.QTextEdit()
        self.OutputAddress_textbox.setFixedSize(480, 80)
        self.OutputAddress_textbox.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-radius: 10px;")
        self.OutputAddress_textbox.setText("")
        
        # Create output address label and text box
        output_address_label1 = QtWidgets.QLabel("Oringial Tune")
        self.Tune_textbox = QtWidgets.QTextEdit()
        self.Tune_textbox.setFixedSize(100, 48)
        self.Tune_textbox.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-radius: 10px; font-size: 32px;")
        self.Tune_textbox.setText("C")
        
        # Create output address label and text box
        output_address_label2 = QtWidgets.QLabel("T Adjust")
        self.Tune_Adjust_textbox = QtWidgets.QTextEdit()
        self.Tune_Adjust_textbox.setFixedSize(100, 48)
        self.Tune_Adjust_textbox.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-radius: 10px; font-size: 32px;")
        self.Tune_Adjust_textbox.setText("0")
        
        # Create console label and text box
        console_label = QtWidgets.QLabel("Console")
        self.ConsoleOutput_textbox = QTextEdit()
        self.ConsoleOutput_textbox.setFixedSize(628,600)
        self.ConsoleOutput_textbox.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-radius: 10px;")
        self.ConsoleOutput_textbox.setPlainText("Ready to translate, let's do it\n")
        self.ConsoleOutput_textbox.setReadOnly(True)
        
        
        button_5 = QtWidgets.QPushButton("Clean")
        button_5.setFixedSize(100,48)
        
        # Add widgets to mid widget layout
        self.mid_widget_layout.addWidget(filename_label, 0, 0)
        self.mid_widget_layout.addWidget(self.filename_textbox, 1, 0)
        self.mid_widget_layout.addWidget(output_address_label, 2, 0)
        self.mid_widget_layout.addWidget(self.OutputAddress_textbox, 3, 0)
        self.mid_widget_layout.addWidget(self.Fix_Output_Address_button, 2, 1)
        self.mid_widget_layout.addWidget(button_1_6, 3, 1)
        
        self.mid_widget_layout.addWidget(output_address_label1, 4, 0)
        #self.mid_widget_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum), 5, 1) # spacer item added
        self.mid_widget_layout.addWidget(output_address_label2, 4, 1)
        self.mid_widget_layout.addWidget(self.Tune_textbox, 5, 0)
        #self.mid_widget_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum), 5, 1) # spacer item added
        self.mid_widget_layout.addWidget(self.Tune_Adjust_textbox, 5, 1)
        self.mid_widget_layout.addWidget(console_label, 6, 0)
        self.mid_widget_layout.addWidget(button_5, 6, 1)
        self.mid_widget_layout.addWidget(self.ConsoleOutput_textbox, 7, 0)
        
        # Add widgets to main widget layout
        main_widget_layout.addWidget(left_widget)
        main_widget_layout.addWidget(self.mid_widget)
        main_widget_layout.addWidget(self.right_widget)
        
        # Set the central widget of the window
        self.setCentralWidget(main_widget)

        # Connect buttons to their respective functions
        button_1.clicked.connect(self.open_file)
        button_1_6.clicked.connect(self.Fix_Output_Address)
        button_2.clicked.connect(self.Start_reco)
        button_3.clicked.connect(lambda : self.Trans_to_Lilypond())
        button_3_5.clicked.connect(self.save_ImgFin)
        button_4.clicked.connect(self.Save_TrainingSet)
        button_5.clicked.connect(self.Clean_print_area)
        button_6.clicked.connect(lambda : self.click_button(4))

        # Set the style sheet for the window and widgets
        self.setStyleSheet("""
            MyUI {
                background-color: rgba(230, 245, 255, 0.7);
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 0.5);
                border-radius: 10px;
                font-size: 22px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.7);
            }
            QListWidget {
                background-color: rgba(255, 255, 255, 0.7);
                }
            """)

#%% TEst
        # 点击按钮的槽函数
    def click_button(self,btn):
        # 遍历右侧布局中的子部件，将其删除
        while self.mid_widget_layout.count():
            child = self.mid_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        text = QtWidgets.QLabel("这是{}的文本".format(btn)) # 创建一个文本部件
        self.mid_widget_layout.addWidget(text) # 将文本部件添加到右侧布局中。
        
        Img = self.Init_Img 
        
        Img = SizeImage(Img)#彩色图片
        global NTC
        
        #notestr = get_text(Img,'num12.1')
        notestr = get_text_Mixed(Img,'chi_sim')
        NTC = get_table(notestr, 0, 0)
        
        (h,w,trivial) = Img.shape
        for i in range(len(NTC)):
            NTC[i][2] = h - int(NTC[i][2])
            NTC[i][4] = h - int(NTC[i][4]) 
        
        Img_Recoed = Draw_Reco(NTC,Img)
        
        MYUI_NP_Plot(self,Img_Recoed)
        
        print(NTC)
        for i in range(len(NTC)):
            if NTC[i][0] == '4':
                Lower_bound = int(NTC[i][4])
                Upper_bound = int(NTC[i][4]) - round(2.2 * (int(NTC[i][2]) - int(NTC[i][4])))
                L_bound = int(NTC[i][1]) - 8
                R_bound = int(NTC[i][3]) + 8
                for i in range(len(NTC)):
                    if NTC[i][0] == '-':
                        if int(NTC[i][1]) > L_bound - 20 and int(NTC[i][3]) < R_bound + 20:
                            if int(NTC[i][2]) < Lower_bound and int(NTC[i][4]) > Upper_bound:
                                print('ooooohhh my -')
                                
                                BaseNum = [2,3,4]
                                for j in range(len(NTC)):
                                    if str.isnumeric(NTC[j][0]):
                                        if int(NTC[j][0]) in BaseNum:
                                            if int(NTC[j][1]) > L_bound and int(NTC[j][3]) < R_bound:
                                                if int(NTC[j][2]) < Lower_bound and int(NTC[j][4]) > Upper_bound:
                                                    print('ooooohhh my number: ',end = '')
                                                    print(int(NTC[j][0]))
                 
                          
    #%% 定义button和UI         
            
    def Start_reco(self):
        po0 = Pool(4)
        po1 = Pool(4)
        
        self.Output_Address = self.OutputAddress_textbox.toPlainText()
        
        # try:
        if 0 == 0:
            GRAND_start = time.time()
            Print_in_UI_1(self, 0, 'START PROCESS', 1, 0)
            Img = self.Init_Img
            path_ori = self.path_ori
            OutImg,Img_Recoed,NTC,path,y1 = Core(self,Img,po0,po1,path_ori)
            self.OutImg = OutImg
            self.NTC = NTC
            self.path = path
            self.y1 = y1
            
            
            self.ImgSized32 = Img_Recoed.copy()
            
            MYUI_NP_Plot(self,Img_Recoed)
            
            content_Collect, Spilt_Method, BeatTime, Tune = Core2(self,self.NTC)
            self.content_Collect = content_Collect
            self.Spilt_Method = Spilt_Method
            self.BeatTime =  BeatTime
            self.Tune =  Tune
            
            Print_in_UI_1(self, GRAND_start, 'All Finished', 1, 1)
        # except:
        #     Print_in_UI_1(self, 0, 'Please select a pircture first', 1, 0)
        
        
        
    def open_file(self):
        fileName,fileType = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), "Image Files(*.jpg *.png)")
        #fileName = full path
       
        slash_locate = str.rfind(fileName,'/')
        self.OrigionImg_Path = fileName
        
        if self.Fix_Output_Address_button.isChecked():
            pass
        else:
            Output_Address_FromfileName = fileName[ : int(slash_locate) ]
            self.OutputAddress_textbox.setPlainText(Output_Address_FromfileName)
        
        path = fileName[int(slash_locate) + 1: ]
        self.path_ori = path
        #path = "Sample_Picture/" + path#Partial path
        
        dot_locate = str.rfind(self.path_ori,'.')
        self.filename_textbox.setPlainText(self.path_ori[0 : int(dot_locate)] + '_Trans')
        
        Img = cv2.imread(fileName)
        self.Init_Img = Img
        
        lbl = QLabel(self)
        pixmap = QPixmap(fileName)  # 按指定路径找到图片
        lbl.setPixmap (pixmap)  # 在label上显示图片
        lbl.setScaledContents (True)  # 让图片自适应label大小
        
        item3 = QListWidgetItem()
        item3.setSizeHint(QSize(600, 800))  # 设置QListWidgetItem大小
        widget = QWidget()
        layoutA = QHBoxLayout()
        layoutA.addWidget(lbl)
        widget.setLayout(layoutA)

        self.right_widget.addItem(item3)
        self.right_widget.setItemWidget(item3, widget)
        
        Print_in_UI_1(self, 0, '[ ' + self.path_ori + ' ]' + '\nFilepath Fixed:\n' + fileName, 1, 0)
        
        
    def Fix_Output_Address(self):
        folderPath = QFileDialog.getExistingDirectory(self, "选取文件夹", os.getcwd())#fileName = full path
        #self.OrigionImg_Path = folderPath
        self.OutputAddress_textbox.setPlainText(folderPath)
        self.Fix_Output_Address_button.setChecked(True)
    
    
    def save_ImgFin(self):
        
        self.Output_Address = self.OutputAddress_textbox.toPlainText()
        Print_in_UI_1(self, 0, 'Getting Progress Image', 1, 0)
        try:
            path_ori = self.path_ori
            path2 = path_ori.split(".")
            propath = self.Output_Address + '/CaseV1A_' + str(Name_No) + '_' + path2[0] + ".jpg"
            Image.fromarray(self.ImgFin_Bin_0).save(propath)

            propath = self.Output_Address + '/CaseV1B_' + str(Name_No) + '_' + path2[0] + ".jpg"
            Image.fromarray(self.ImgFin_Bin_1).save(propath)
            
            propath = self.Output_Address + '/CaseV1F_' + str(Name_No) + '_' + path2[0] + ".jpg"
            Image.fromarray(self.ImgFin_Recoed_0).save(propath)
            
            propath = self.Output_Address + '/CaseV1G_' + str(Name_No) + '_' + path2[0] + ".jpg"
            Image.fromarray(self.ImgFin_Recoed_1).save(propath)
            
            Print_in_UI_1(self, 0, 'Progress Saved', 1, 0)
            
            
            
        except:
            Print_in_UI_1(self, 0, 'Please import image and extarct it first', 1, 0)
                
        
    def Save_TrainingSet(self):
        
        self.Output_Address = self.OutputAddress_textbox.toPlainText()
        Print_in_UI_1(self, 0, 'Getting training data', 1, 0)
        try:
            path_ori = self.path_ori
            path2 = path_ori.split(".")
            propath = self.Output_Address + "/TrainSampleV8_Beta_" + path2[0] + ".tif"
            Image.fromarray(self.OutImg).save(propath)
            Print_in_UI_1(self, 0, 'Saved!', 1, 0)
        except:
            Print_in_UI_1(self, 0, 'Please process a pircture first', 1, 0)
    
    
    def Trans_to_Lilypond(self):
        Core3(self, self.content_Collect, self.Spilt_Method, self.BeatTime, self.Tune)
        Print_in_UI_1(self, 0, 'Saved to: [ ' + self.ly_path + ' ]', 1, 0)
        

    def tab1UI(self):
        layout = QFormLayout()
        layout.addRow("参数1", QLineEdit())
        layout.addRow("参数2", QLineEdit())
        self.tab1.setLayout(layout)


    def Clean_print_area(self):
        self.ConsoleOutput_textbox.setPlainText("Ready to translate, let's do it\n")
        
    def initUI(self):
        pass


#%% UI以外的主程序
#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#%%  tesseract ocr
#Source： https://github.com/UB-Mannheim/tesseract/wiki
os.environ['TESSDATA_PREFIX']='C:/Program Files/Tesseract-OCR/tessdata'
base_folder = os.path.join("C:/Program Files","Tesseract-OCR")

tesseract_exe = os.path.join(base_folder, "tesseract.exe")
tessdata_dir_config = os.path.join(base_folder, "tessdata")
pytesseract.tesseract_cmd = tesseract_exe


def get_grayscale(image):
    return cv2.cvtcolor(image, cv2.cOLOR_bGR2GRaY)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_bINaRY + cv2.THRESH_OTSU)[1]


def canny(image):
    return cv2.canny(image, 100, 200)


def cut_image(image):
    cropped = image[image.shape[0] // 5:image.shape[0] - image.shape[0] // 5, 0:image.shape[1]]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("cv_cut_thor.png", cropped)


def get_text(image,OCR_model):
    text = pytesseract.image_to_boxes\
        (image,config=\
         '--psm 13  -c tessedit_char_whitelist=12345670-X.()^ --dpi 300 --oem 1'\
             ,lang=OCR_model)
    return text


def get_text_Mixed(image,OCR_model):
    # OCR_model = 'eng'
    # image = cv2.imread(image_path)
    # gray = get_grayscale(image)
    # thresh = thresholding(gray)
    # canny_result = canny(thresh)

    #text = pytesseract.image_to_boxes(image,config=' --psm 7',lang="num5")
    #text = pytesseract.image_to_boxes(image,config=' --psm 13 -c tessedit_char_whitelist=12345670-X.()',lang=OCR_model)
    text = pytesseract.image_to_boxes\
        (image,config=\
         '--psm 11 -c tessedit_char_blacklist=1234567890-X.()^ --dpi 300 --oem 0'\
             ,lang=OCR_model)
    return text


#%% 以下与ts ocr无关
def MYUI_NP_Plot(self,Img_Recoed):
    try:
        Img = Img_Recoed
     
        Img_QImage = QImage(Img, Img.shape[1], Img.shape[0], Img.shape[1]*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(Img_QImage)#.scaled(label_width, label_height)
    
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)  # 在label上显示图片
        lbl.setScaledContents (True)  # 让图片自适应label大小
        
        item3 = QListWidgetItem()
        item3.setSizeHint(QSize(600, 800))  # 设置QListWidgetItem大小
        widget = QWidget()
        layoutA = QHBoxLayout()
        layoutA.addWidget(lbl)
        widget.setLayout(layoutA)
    
        self.right_widget.addItem(item3)
        self.right_widget.setItemWidget(item3, widget)
    except:
        print('No app, direct print it')
        
    
    
#%% UI部分结束     
def Rotate_Img0(imagea,ImgSized3):
    imagea = cv2.GaussianBlur(imagea,(3,3),1.5)
    #Imshow_CV2('kkk',imagea)
    imagea = BinaryOperation(imagea, 180)
    Imshow_CV2('kkk',imagea)
    # imagea = cv2.medianBlur(imagea,3)                             
    # imagea = BinaryOperation(imagea, 180)
    
    #edge = cv2.Canny(imagea, 100,125,apertureSize=3)
    #edge = cv2.Canny(imagea, 60,65,apertureSize=5)
    edge = 255 - imagea
    
    minLineLength = 25 #10 # height/32# 最低线段的长度，小于这个值的线段被抛弃
    maxLineGap = 125 #120 # height/40# 线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
    rho = 1.0
    theta = np.pi/540
    threshold = 30
    lines = cv2.HoughLinesP(edge, rho, theta,  threshold, minLineLength, maxLineGap)
    
    minLineLength = 50 #10 # height/32# 最低线段的长度，小于这个值的线段被抛弃
    maxLineGap = 250 #120 # height/40# 线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
    lines = cv2.HoughLinesP(edge,1, np.pi/180, 10, minLineLength, maxLineGap)


    NoLineImgArray = imagea.copy()
    if lines.all == None:
        print('\nTo this image, no horizontal line can be detected by Hough')
    else:
        for i in range(len(lines)):
            line = lines[i]
            x1 = line[0,0]
            y1 = line[0,1]
            x2 = line[0,2]
            y2 = line[0,3]
            if abs( (y1 - y2) / (x1 - x2) ) > 200 :#去除竖线
                x1 = 0
                x2 = len(imagea[0])
                y1 += 8
                y2 += 8
                cv2.line(NoLineImgArray, (x1, y1), (x2, y2), (255,255,255), 18)
                cv2.line(ImgSized3, (x1, y1), (x2, y2), (225,125,155), 18)
            #cv2.line(NoLineImgArray, (x1, y1), (x2, y2), (255,255,255), 15)
    # NoLineImg = (np.uint8(NoLineImgArray));

    Imshow_CV2('Recoed',ImgSized3)
    
    
    # print(rotate_angle)

    # img = 255 - img#part1,避免黑边
    # rotate_img = ndimage.rotate(img, rotate_angle, reshape=True)
    
    # rotate_angle = abs(rotate_angle)
    
    #h = np.size(rotate_img,0)
    #w = np.size(rotate_img,1)
    #h_cut = math.floor( w*math.sin( math.radians(rotate_angle)) )
    #w_cut = math.floor( h*math.sin( math.radians(rotate_angle)) )
    #rotate_img = rotate_img[h_cut: h - h_cut, w_cut: w - w_cut]
    
    rotate_img = 2
    #rotate_img = 255 - rotate_img #part2，避免黑边
    return rotate_img

    #%%    
def Rotate_Img(img):
    Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 先要转换为灰度图片
    ret, gray = cv2.threshold(Grayimg, 150, 255,cv2.THRESH_BINARY) # 这里的第二个参数要调，是阈值

    edge = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 25 #10 # height/32# 最低线段的长度，小于这个值的线段被抛弃
    maxLineGap = 90 #120 # height/40# 线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
    lines = cv2.HoughLines(edge,1, np.pi/180, 100, minLineLength, maxLineGap)

    rotate_angle_Collect = []
    Pos_Neg = []
    for i in range(len(lines)):
        for rho,theta in lines[0]: 
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
        if x1 == x2 or y1 == y2:
            pass
        else:
            angle = float(y2-y1)/(x2-x1)
            rotate_angle = round(math.degrees(math.atan(angle)),8)
            if rotate_angle > 45:
                rotate_angle = -90 + rotate_angle
            elif rotate_angle < -45:
                rotate_angle = 90 + rotate_angle
            
            if rotate_angle > 0:
                Pos_Neg.append(1)
            else:
                Pos_Neg.append(-1)
                rotate_angle = -rotate_angle
            rotate_angle_Collect.append(rotate_angle)
    
    if rotate_angle_Collect == []:
        rotate_img = img
    else:
        i_max = max(np.bincount(rotate_angle_Collect)) - 1 
        rotate_angle = Pos_Neg[ i_max ] * rotate_angle_Collect[ i_max ]#取出众数
        print(rotate_angle)
        
       
        img = 255 - img#part1,避免黑边
        rotate_img = ndimage.rotate(img, rotate_angle, reshape=True)
        
        rotate_angle = abs(rotate_angle)
        #h = np.size(rotate_img,0)
        #w = np.size(rotate_img,1)
        #h_cut = math.floor( w*math.sin( math.radians(rotate_angle)) )
        #w_cut = math.floor( h*math.sin( math.radians(rotate_angle)) )
        #rotate_img = rotate_img[h_cut: h - h_cut, w_cut: w - w_cut]
        rotate_img = 255 - rotate_img #part2，避免黑边
    return rotate_img
    

#%% Rotate结束
def get_table(notestr,i_div,i_bar):
    lista = notestr.strip(' ').split(' ')

    n = 0
    m = int((len(lista) + (len(lista)-1)/5))
    listb = [0]*m
    for i in range(len(lista)):
        listb[i+n] = lista[i]
        if i%5 == 0 and i > 0 and i < len(lista)-1:
            listclip = lista[i]
            listb[i+n] = listclip[0]
            listb[i+n+1] = listclip[2]
            n = n + 1   
    
    del(listb[-1])
    
    try:
        listb[-1] = 0
        listc = np.array(listb)
        notetable = listc.reshape(int(len(listb)/6),6)
        
        a = [i_div for _ in range(len(notetable))]
        a = np.array(a)
        a2 = [i_bar for _ in range(len(notetable))]
        a2 = np.array(a2)
        
        b = np.transpose(notetable)
        ab = np.vstack( (b,a,a2) )
        notetable = np.transpose(ab)
    except:
        print('notetable be empty here')
        notetable = ['?',0,0,0,0,0]
    return notetable

#%% 原本
def BinaryOperation0(image,
                    threshold: int):  
    Grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 先要转换为灰度图片
    Imshow_CV2('Garyimg',Grayimg)
    ret, thresh = cv2.threshold(Grayimg, 150, threshold,cv2.THRESH_BINARY) # 这里的第二个参数要调，是阈值
    Imshow_CV2('Garyimg',thresh)
    return thresh


#%% 改动
def BinaryOperation2(image,
                    threshold: int):  
    h = len(image)
    w = len(image[0])
    Gray_Img = np.zeros([h,w])
    Gray_Img = np.uint8(Gray_Img)
    Gray_Img = BinaryOperation2_core(h,w,Gray_Img,image,threshold)
    return Gray_Img

@jit(nopython=True)
def BinaryOperation2_core(h,w,Gray_Img,image,threshold):
    for y in range(h):
        for x in range(w):
            B = int(image[y][x][0])
            G = int(image[y][x][1])
            R = int(image[y][x][2])
            if B > threshold or G > threshold or R > threshold:
                pixel = 255
            else:
                color_bias = max(B,G,R) - min(B,G,R)
                color_sum = B + G + R

                if color_bias < 78 and color_sum < 220:#总体深色且偏差不大
                    pixel = 0
                elif color_bias < 62 and color_sum < 290:#总体深色且偏差不大
                    pixel = 0
                elif color_bias < 52 and color_sum < 320:#较暗，蛮灰
                    pixel = 0
                elif color_bias < 42 and color_sum < 350:#较暗，蛮灰
                    pixel = 0
                elif color_bias < 32:#不偏色，很灰
                    pixel = 0
                else:#不但不深，还偏色
                    pixel = 255

            Gray_Img[y][x] = pixel
    return Gray_Img


#%% 改动
def BinaryOperation3(image,
                    threshold: int):  
    h = len(image)
    w = len(image[0])
    Gray_Img = np.zeros([h,w])
    Gray_Img = np.uint8(Gray_Img)
    Gray_Img = BinaryOperation3_core(h,w,Gray_Img,image,threshold)
    return Gray_Img

@jit(nopython=True)
def BinaryOperation3_core(h,w,Gray_Img,image,threshold):
    for y in range(h):
        for x in range(w):
            B = int(image[y][x][0])
            G = int(image[y][x][1])
            R = int(image[y][x][2])
            if B > threshold or G > threshold or R > threshold:
                pixel = 255
            else:
                color_bias = max(B,G,R) - min(B,G,R)
                color_sum = B + G + R

                if color_bias < 78 and color_sum < 320:#总体深色且偏差不大
                    pixel = 0
                elif color_bias < 62 and color_sum < 350:#总体深色且偏差不大
                    pixel = 0
                elif color_bias < 52 and color_sum < 390:#较暗，蛮灰
                    pixel = 0
                elif color_bias < 42 and color_sum < 460:#较暗，蛮灰
                    pixel = 0
                elif color_bias < 32:#不偏色，很灰
                    pixel = 0
                else:#不但不深，还偏色
                    pixel = 255

            Gray_Img[y][x] = pixel
    return Gray_Img


#%% 
def SizeImage(image):
    try:
        (h,w,trivial) = image.shape
    except:
        print('Please Select a correct image path')
    print('\nOriginal Size:   ',end = '')
    print(h,w)
    
    wmax = 3000
    wmin = 2800
    while w < wmin or wmax < w:
        if w < wmin:
            w = int(w * 1.25)
            h = int(h * 1.25)
        if w > wmax:
            w = int(w / 1.2)
            h = int(h / 1.2)
    h = h + h%2
    w = w + w%2
    ImgSized = cv2.resize(image,(w,h))
    print('Normalized Size: ',end = '')
    print(h,w) 
    return ImgSized


def Draw_Reco(notetable,
              image):
    listlen = len(notetable)
    for i in range(listlen):
        if notetable[i][0] != '|':
            y1 = int(notetable[i,3]) 
            y2 = int(notetable[i,1]) 
            x1 = int(notetable[i,4]) 
            x2 = int(notetable[i,2]) 
            RawNum = str(notetable[i,0])
            if str.isnumeric(RawNum) or RawNum == 'X':
                [B,G,R] = 230,60,90
                T = 2
            elif RawNum == '-' or RawNum == '.':
                [B,G,R] = 60,90,230
                T = 2
            elif RawNum == '^':
                [B,G,R] = 60,200,230
                T = 3
            elif RawNum == '?':
                [B,G,R] = 240,200,240
                T = 4
            else:
                [B,G,R] = 90,230,60
                T = 2
            cv2.rectangle(image, (y1,x1),(y2,x2),(B, G, R), T)
            cv2.putText(image, str(notetable[i,0]), (y1,x2), cv2.FONT_HERSHEY_COMPLEX, 2, (B, G, R), 3)
    return image


def Draw_Reco_Blankit(notetable,
              image_b):
    listlen = len(notetable)
    for i in range(listlen):
        y1 = int(notetable[i,3]) 
        y2 = int(notetable[i,1]) 
        x1 = int(notetable[i,4]) 
        x2 = int(notetable[i,2]) 
        cv2.rectangle(image_b, (y1,x1),(y2,x2),(240, 240, 240), -1)
    return image_b
    

def Caculate_RowPix(ImgSized):  
    ImgSized01 = np.where(ImgSized==0,1,0)
    ImgSized01 = ImgSized01[:,round(np.size(ImgSized01,1)*0.02):round(np.size(ImgSized01,1)*0.98)]#取中间部分,避免边界干扰
    pixel_num = np.sum(ImgSized01,axis = 1)
    return pixel_num


def Caculate_ColPix(ImgSized01):
    ImgSized01 = ImgSized01[:, 0:np.size(ImgSized01,1)]#取中间部分
    c_num = np.sum(ImgSized01,axis = 0)
    c_num_mean = np.mean( [i for i in c_num if i > 30] )
    Lc = len(c_num)
    for i in range(Lc):
        if c_num[i] > c_num_mean:
            break
    Left_Start = i
    for i in range(Lc):
        if c_num[Lc - i - 1] > c_num_mean:
            break
    Right_Start = Lc - i - 1
    return Left_Start,Right_Start


def del_SmallConnectedRegion(Img, threshold):
    background= 0
    image_array = np.uint8(255 - Img)
    labeled_image, num = measure.label(image_array, connectivity=2, background=background, return_num=True)#使用四联通，因为数字不会被误伤
    regions = measure.regionprops(labeled_image)
    Vertical_Pixel_half = int(0.5 * len(image_array))
    Horizontal_Pixel_half = int(0.5 * len(image_array[0]))
    
    R = []
    for i in range(len(regions)):
        R.append(int(regions[i].area))
    R = np.array(R)
    ImgSized_Del = dSCR_core(labeled_image, R, Vertical_Pixel_half, Horizontal_Pixel_half, threshold, image_array)
    ImgSized_Del = np.uint8(ImgSized_Del)
    return ImgSized_Del

@jit(nopython=True)
def dSCR_core(labeled_image, R, Vertical_Pixel_half, Horizontal_Pixel_half, threshold, image_array):
    for y_half in range(Vertical_Pixel_half):
        y = y_half*2
        if sum(labeled_image[y,:]) == 0:
            continue
        else:
            for x_half in range(Horizontal_Pixel_half):
                x = x_half*2
                region_num = labeled_image[y, x]
                if region_num != 0 and (R[region_num - 1] < threshold):
                    if image_array[y, x] == 255:
                        image_array[y, x] = 0
                    if image_array[y + 1, x] == 255:
                        image_array[y + 1, x] = 0
                    if image_array[y, x + 1] == 255:
                        image_array[y, x + 1] = 0
                    if image_array[y + 1, x + 1] == 255:
                        image_array[y + 1, x + 1] = 0
                        
    ImgSized_Del = 255 - image_array
    return ImgSized_Del
    
    
#%% 原版本
def RemoveLines1(imagea):
    edge = cv2.Canny(imagea, 120,195,apertureSize=3)
    minLineLength = 25 #10 # height/32# 最低线段的长度，小于这个值的线段被抛弃
    maxLineGap = 90 #120 # height/40# 线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
    lines = cv2.HoughLinesP(edge,1, np.pi/180, 100, minLineLength, maxLineGap)
    NoLineImgArray = imagea.copy()
    if lines.all == None:
        print('\nTo this image, no horizontal line can be detected by Hough')
    else:
        for i in range(len(lines)):
            line = lines[i]
            x1 = line[0,0]
            y1 = line[0,1]
            x2 = line[0,2]
            y2 = line[0,3]
            if x1 == x2:#去除竖线
                x1 = 0
                x2 = 1
                y1 = 0
                y2 = 1
            if y1 == y2:
                x1 = 0
                x2 = len(imagea[0])
            cv2.line(NoLineImgArray, (x1, y1), (x2, y2), (255,255,255), 18)
    NoLineImg = (np.uint8(NoLineImgArray));
    NoLineImg01 = np.where(NoLineImg == 0,1,0)
    return NoLineImg,NoLineImg01


def Enhance_BarLine(Img01, Img):
    Img = np.array(Img)
    background= 255
    image_array = np.uint8(Img01)
    labeled_image, num = measure.label(image_array, connectivity=2, background=background, return_num=True)#使用四联通，因为数字不会被误伤
    regions = measure.regionprops(labeled_image)
    Vertical_Pixel_half = int(0.5 * len(image_array))
    Horizontal_Pixel = len(image_array[0])
    
    R = []
    H = []
    W = []
    for i in range(len(regions)):
        R.append(int(regions[i].area))
        H.append(int(regions[i].bbox[2]) - int(regions[i].bbox[0]))
        W.append(int(regions[i].bbox[3]) - int(regions[i].bbox[1]))
    R = np.array(R)
    H = np.array(H)
    W = np.array(W)
    ImgSized_Enh = EBL_core(labeled_image, Img, H, W, Vertical_Pixel_half, Horizontal_Pixel, image_array)
    ImgSized_Enh = np.uint8(ImgSized_Enh)
    #Imshow_CV2('BarLineEnhance', ImgSized_Enh)
    return ImgSized_Enh

@jit(nopython=True)
def EBL_core(labeled_image, Img, H, W, Vertical_Pixel_half, Horizontal_Pixel, image_array):
    for y_half in range(Vertical_Pixel_half):
        y = y_half*2
        C = [0,0,0]
        if sum(labeled_image[y,:]) == 0:
            continue
        else:
            for x in range( int(Horizontal_Pixel) ):
                region_num = labeled_image[y, x]
                if region_num != 0 and (2 < W[region_num - 1] < 22) and (88 < H[region_num - 1] < 188):
                #if region_num != 0 and (1 < H[region_num - 1] < 12) and (38 < W[region_num] < 308):#可以横线
                    Img[y, x] = C
                    Img[y + 1, x] = C

    ImgSized_Del = Img
    return ImgSized_Del

#%%
def Plot_DivRowOutcome(Plot_name,r_distance,pixel_num,l_distance,DelMark,path_ori):
    plt.figure()
    plt.title(path_ori + ': ' + Plot_name)
    if len(r_distance) > 10:
        x = list(range(len(r_distance)))
        plt.plot(x,r_distance, color='b', label='label1', linewidth=0.8)
    if len(pixel_num) > 10:
        x = list(range(len(pixel_num)))
        plt.plot(x,pixel_num, color='r', label='label1', linewidth=0.8)
    if len(l_distance) > 10:
        x = list(range(len(l_distance)))
        plt.plot(x,l_distance, color='g', label='label1', linewidth=0.8)#右边的距离
    if len(DelMark) > 10:
        x = list(range(len(DelMark)))
        plt.plot(x,DelMark, color='y', label='label1', linewidth=0.8)
    plt.show 
    return 0


def Imshow_CV2(windowname,Img):
    cv2.namedWindow(windowname,0)
    cv2.imshow(windowname,Img)
    cv2.waitKey(0)
    
    
def PLTshow(windowname,Img,Compress_rate):
    (h,w) = Img.shape
    Img= cv2.resize(Img,(round(w*Compress_rate), round(h*Compress_rate)))
    plt.figure(windowname)
    plt.imshow(Img,cmap='gray')
    plt.title(windowname)
    plt.imshow
    

@jit(nopython=True)
def Caculate_DistanceFromSide(Img01):
    ss = []
    ss2 = []
    img_width = len(Img01[0])
    for i in range(0,len(Img01)):
        if sum(Img01[i]) == 0:#两行实现明显加速
            s = img_width
            s2 = img_width
        else:
            location_s = np.where(Img01[i] == 1)
            s = img_width - location_s[0][-1]#最后一个为s[0][-1]
            s2 = location_s[0][0]#最后一个为s[0][-1]
        ss.append(s)
        ss2.append(s2)
    r_distance = [-ssx + max(ss) for ssx in ss]#Blue line in the Plot
    l_distance = ss2
    return r_distance,l_distance


def RemoveRows_ThinerThanThreshold(r_distance,l_distance,pixel_num,threshold):
    start = 0
    y1 = []
    y2 = []
    Del_Reco = []
    for i in range(len(r_distance)):
        if start == 0:
            if r_distance[i] > 0:
                start_point = i
                start = 1
        if start == 1:
            if r_distance[i] == 0:
                end_point = i
                start = 0
                if (end_point - start_point) > threshold:#值得注意
                    y1.append(start_point)
                    y2.append(end_point)
                else:
                    Del_Reco.append((end_point - start_point))
                    for j in range(start_point,end_point):
                        r_distance[j] = 0
                        l_distance[j] = 0
                        pixel_num[j] = 0       
    Del_Reco = abs(np.sort( - np.array(Del_Reco) ))
    # print('\n4:Raws [deleted], thiner than xx:')
    # print(Del_Reco)
    return r_distance,l_distance,pixel_num,y1,y2


def Save_LeastRow(r_distance, l_distance, pixel_num, threshold, Right_Start):
    len_rdis = len(r_distance)
    row_start = round(0.75 * len_rdis) 
    for i in range(round(0.75 * len_rdis), len_rdis - 1):
        if abs(r_distance[i] - r_distance[i + 1]) > 8:
            row_end = i 
            row_len = row_end - row_start 
            if row_len > threshold:
                if r_distance[round(0.5 * (row_end + row_start))] > 8:
                    print(row_start)
                    print(row_end)
                    print('OKKKKKK')
                    for m in range(row_start,row_end):
                        r_distance[m] = Right_Start
            row_start = i
    return r_distance,l_distance,pixel_num
    
    
def RemoveRows_DistinctivelyFarFromRight(r_distance,l_distance,pixel_num,repeat,redundancy):
    #第二次算平均，去除不够靠右的
    r_distance_none0=[]
    for n in range(repeat):
        r_distance_none0 = [j for j in r_distance if j > 0]#先除去所有零
        avg_distance = np.mean(r_distance_none0)#再算出平均的高度，用于去除以下的部分
        for i in range(0,len(r_distance)):
            if r_distance[i] < avg_distance - redundancy:
                r_distance[i] = 0
                l_distance[i] = 0
                pixel_num[i] = 0
    return r_distance,l_distance,pixel_num


def Abbrivate_ByPixel(y1,y2,r_distance,pixel_num):
    avg_pixel_num = []
    for i in range(len(y1)):
        section_pixel_num = np.array(pixel_num[y1[i]:y2[i]])
        section_pixel_num = [j for j in section_pixel_num if j > 10]#去除过小的值
        section_pixel_num = np.sort(section_pixel_num)
        section_pixel_num = section_pixel_num[0:round(len(section_pixel_num)*0.75)]#去除那些最大值
        
        avg_num = np.mean(section_pixel_num)
        avg_pixel_num.append(avg_num)
        for j in range(y1[i],y2[i]):#先把左边的边界缩进,去除过少的红色，More
            if pixel_num[j] < avg_num + 12:
                r_distance[j] = 0
                pixel_num[j] = 0
            else:
                y1[i] = j - 4#稍微补一点点
                break
        for j in range(y1[i],y2[i]):#再把右边的边界缩进
            if pixel_num[y2[i] + y1[i] - j] < avg_num :
                r_distance[y2[i] + y1[i] - j] = 0
                pixel_num[y2[i] - j] = 0
            else:
                y2[i] = y2[i] + y1[i] - j + 4
                break
    i_adjust = 0#保证删除结果与图像绘制一样
    for i in range(len(y1)):#再次去除过矮的行
        i = i - i_adjust
        Raw_Content = np.mean( (r_distance[int(y1[i]):int(y2[i])]) )
        if Raw_Content < max(r_distance)*0.6:
            y1 = np.delete(y1,i)
            y2 = np.delete(y2,i)
            i_adjust += 1#保证删除后仍在指定位置 
    return y1,y2,r_distance,pixel_num,avg_pixel_num


def RemoveRaws_ThinerThanAvg(y1,y2):
    for j in range(2):
        Row_Heights = np.array(y2) - np.array(y1)
        Row_Height = np.mean(Row_Heights)
        i_Adjust = 0
        for i in range(len(y1)):
            if Row_Heights[i] < Row_Height * 0.8:
                i = i - i_Adjust
                print('\n9: Tiny Line Adjusted.')
                y1 = np.delete(y1,i)
                y2 = np.delete(y2,i)
                i_Adjust += 1
    return y1,y2


def RemoveRows_DetcetPossibleWriter(Right_Start,Left_Start,Img,y1,y2,avg_pixel_num):
    #用于去除可能是作曲家名的第一个识别行
    ImgSized01 = Img
    Raw_Len = Right_Start - Left_Start
    avg_pixel_num_ori = avg_pixel_num
    for i in range( round(len(avg_pixel_num) * 0.4) ):#去除一些最大值
        avg_pixel_num.remove(max(avg_pixel_num))
    avg_pixel_num_mean = np.mean(avg_pixel_num)
    remove_sum = 0
    del_location_distance = 0#避免删除掉与作曲家有距离的模糊第一行
    first_del_location = y2[0]
    for j in range(3):#多次检查
        Row_Sums = []
        for i in range(y1[0],y2[0]):
            ImgSized01_section = ImgSized01[i,Left_Start + round(0.28*(Raw_Len)):Right_Start - round(0.28*(Raw_Len))]#取中间部分,避免边界干扰
            Row_Sums.append( np.sum(ImgSized01_section) )
        if avg_pixel_num_ori[0] + np.mean(Row_Sums) < avg_pixel_num_mean:
            remove_sum = remove_sum + ( y2[0] - y1[0] )
            if remove_sum < 85:#当第一行乐谱太空的时候，避免删去太多
                del_location_distance = y1[0] - first_del_location
                if del_location_distance < 128:
                    print('\n8: Writer Line Adjusted.')
                    y1 = np.delete(y1,0)
                    y2 = np.delete(y2,0)
    return y1,y2


def Adjust_LineChartIndex(r_distance,pixel_num,y1,y2):
    for i in range(0,len(r_distance)):
        j = 0
        if i > y2[j]:
            j += 1
        if j == len(y1) or i < y1[j]:
            r_distance[i] = 0
            pixel_num[i] = 0
    return r_distance,pixel_num


#%%
def SpiltWhole_ToBar(Img,y1,y2,i,spilt_Collect,Left_Start):
    #print('[' + str(i+1) + ']:')
    y_Height = y2[i] - y1[i]
    section = Img[y1[i] - round(y_Height*0.24): y2[i] + round(y_Height*0.24), 0: np.size(Img,1)]
    y_Height_enlarged = y2[i] - y1[i] + 2*round(y_Height*0.18)
    
    #Imshow_CV2('A row' , section)
    
    background= 0
    image_array = 255 - section
    labeled_image, num = measure.label(image_array, connectivity=2, background=background, return_num=True)#使用四联通，因为数字不会被误伤
    regions = measure.regionprops(labeled_image)
    x1 = [0]
    x2 = [(Left_Start - 20)]#定义左端开始
    temp_x1 = 0
    scan_row = round(y_Height * 0.5)
    for x in range(len(image_array[0])):#在中间行进行扫描，找到纵向贯穿的竖线分小节
        spilt =  0
        if labeled_image[scan_row, x] != 0:
            Lbound = regions[labeled_image[scan_row, x] - 1].bbox[1]
            Rbound = regions[labeled_image[scan_row, x] - 1].bbox[3]
            Ubound = regions[labeled_image[scan_row, x] - 1].bbox[0]
            Dbound = regions[labeled_image[scan_row, x] - 1].bbox[2]
            if  regions[labeled_image[scan_row, x] - 1].bbox[2] - regions[labeled_image[scan_row, x] - 1].bbox[0] >= y_Height_enlarged - 5:
                if x1[-1] != Lbound and temp_x1 != Lbound:#防止重复操作
                    width_barline = Rbound - Lbound
                    if width_barline > 12:
                        V_Mid = round((Ubound + Dbound) * 0.5)
                        Check = int(image_array[V_Mid , (Lbound + 2)]) + int(image_array[V_Mid , (Rbound - 2)])
                        if Check == 510:#两处都是黑色像素
                            spilt = 1
                            #print('Seems a board | has been dectetd and cause a split')
                        else:
                            spilt = 0
                            #print('Seems a tall ( has been dected and ignored from spilt')
                            temp_x1 = Lbound
                    else:
                        spilt = 1
        if spilt == 1:#检查小节是否过短，若是则延伸
            if (Lbound - x2[-1]) < 68:
                spilt = 0
                #print('Seems a || or strange deformity has been dected and cause an enlarge split: ',end = '')
                #print(Lbound - x2[-1])
                x2 = x2[0:-1]
                x2.append( Rbound )
                
                if type(spilt_Collect) == list:#防止只有一个时报错
                    spilt_Collect[3] = Rbound
                else:
                    spilt_Collect[-1][3] = Rbound
                temp_x1 = Lbound
                
        if spilt == 1:#最终执行
            x1.append( Lbound )
            x2.append( Rbound )
            if i == 0:
                spilt_Collect = ['|',Lbound,Ubound+y1[i],Rbound,Dbound+y1[i],0]
            else:
                spilt_Collect = np.vstack( (spilt_Collect,['|',Lbound,Ubound+y1[i],Rbound,Dbound+y1[i],0]))
    return x1,x2,spilt_Collect


#%%
def Draw_LocatedBars(y1,y2,X12s,ImgSized3):
    layer = np.zeros(ImgSized3.shape, np.uint8)
    for i in range(len(X12s)):
        X12sec = X12s[i]
        #yes = X12sec[2]
        x1 = X12sec[0]
        x2 = X12sec[1]
        #layer = layer + 255
        for j in range(len(x1) - 1):
            cv2.rectangle(layer, (x2[j] + 8,y1[i] - 3),( x1[j+1] - 8,y2[i] + 3),(120, 200, 80), -1)
    ImgSized3 = cv2.addWeighted(ImgSized3, 0.7, layer, 0.95, 1)
    #ImgSized3 = cv2.addWeighted(layer, 1.0, ImgSized3,0.8, 0)
    #Imshow_CV2('ttt',ImgSized3)
    return ImgSized3


def Creat_Layer(LineLocation,ImgSized32, RGB, layer):
    for i in range(np.size(LineLocation,0)):
        y1 = LineLocation[i,1]
        y2 = LineLocation[i,2]
        x1 = LineLocation[i,3]
        x2 = LineLocation[i,4]
        cv2.rectangle(layer, (x2,y1),(x1,y2),(RGB[0], RGB[1], RGB[2]), -1)
    return layer
    
    
def Spilt_and_Clean(y1,y2,X12s,ImgSized,i,Extra_Edge_len):
    X12sec = X12s[i]
    yes = X12sec[2]
    x1 = X12sec[0]
    x2 = X12sec[1]
    BIDs = []
    Bars = []
    
    if yes == 1:
        for j in range(len(x1) - 1):
            section_onebar = ImgSized[y1[i] - 3: y2[i] + 3, x2[j] + 2: x1[j+1] - 2]
            section_onebar = Extend_Onebar(section_onebar,Extra_Edge_len,x1,x2,j,len(ImgSized[0]))
            Bars.append(section_onebar)   
        
        for j in range(len(x1) - 1):
            section_onebar = Clean_OneBar(Bars[j],Extra_Edge_len,y1,y2,i,j)  
            Info = [i + 1, j + 1, section_onebar, y1[i], x2[j], 1]
            BIDs.append(Info)

            if j == 0:
                section_onerow = section_onebar
            else:
                section_onerow = np.hstack( (section_onerow,section_onebar) )
    return BIDs,section_onerow


def Clean_OneBar(section_onebar,Extra_Edge_len,y1,y2,i,j):
    threshold = 280
    background= 0
    fill_value = 0
    image_array = 255 - section_onebar
    labeled_image, num = measure.label(image_array, connectivity=2, background=background, return_num=True)#使用四联通，因为数字不会被误伤
    regions = measure.regionprops(labeled_image)
    for x_half in range( int(np.floor(0.5 * len(image_array[0]))) ):#通过查看上下是否连接顶底，判断是噪声还是数字
        x = x_half*2
        if sum(image_array[:,x]) == 0:
            pass
        else:
            for y_half in range(int(np.floor(0.5 * len(image_array)))):
                y = 2*y_half
                noise = 0
                Cleaning = 0
                if labeled_image[y, x] != 0:         
                    if regions[labeled_image[y, x] - 1].area < threshold:
                        Cleaning = 1
                    else:
                        region_length = ( regions[labeled_image[y, x] - 1].bbox[3] - regions[labeled_image[y, x] - 1].bbox[1] )
                        region_height = ( regions[labeled_image[y, x] - 1].bbox[2] - regions[labeled_image[y, x] - 1].bbox[0] )
                        if region_length > 42 and region_height < 22:#太细长的也直接删
                            Cleaning = 1
                        
                    if Cleaning == 1:
                        if regions[labeled_image[y, x] - 1].bbox[0] < Extra_Edge_len + 2:
                            noise = noise + 1
                        if regions[labeled_image[y, x] - 1].bbox[2] > (y2[i]-y1[i]) + Extra_Edge_len:
                            noise = noise + 1
                if noise == 1:
                    image_array[y, x] = fill_value
                    image_array[y + 1, x] = fill_value
                    image_array[y, x + 1] = fill_value
                    image_array[y + 1, x + 1] = fill_value
    section_onebar = 255 - image_array
    return section_onebar

#@jit(nopython=True)
def Extend_Onebar(section_onebar,Extra_Edge_len,x1,x2,j,pic_width):
    Extra_Edge = np.full( (Extra_Edge_len,len(section_onebar[0])),255,dtype='uint8' )
    section_onebar = np.vstack( (Extra_Edge,section_onebar) )
    section_onebar = np.vstack( (section_onebar,Extra_Edge) )
    Extra_Edge = np.full( (len(section_onebar),x2[j+1] - x1[j+1]),255,dtype='uint8' )
    section_onebar = np.hstack( (section_onebar,Extra_Edge) )
    if j == len(x1) - 2:
        Ending_Extra_Edge_len = pic_width - x2[j+1]
        Extra_Edge = np.full( (len(section_onebar),Ending_Extra_Edge_len),255,dtype='uint8' )
        section_onebar = np.hstack( (section_onebar,Extra_Edge) )
    return section_onebar


#%% 
def TextReco_And_AddToNotetable(section_onebar,y1e,x2e,Extra_Edge_len,row_num,bar_num,OCR_model,plotbar):
    #Imshow_CV2('rrrr', section_onebar)
    notestr = get_text(section_onebar,OCR_model)
    if len(notestr) == 0:
        print('Warning! A bar seems be empty?:',end = '')
        bar_code = (row_num)*1000 + (bar_num)
        print(bar_code)
        notetable3 = ['?',0,0,0,0,0,0,0]
    else:
        notetable3 = get_table(notestr, row_num, bar_num)#实现完美换小节
        for k in range(len(notetable3)):
            notetable3[k][2] = int(notetable3[k][2]) + int(y1e) - Extra_Edge_len
            notetable3[k][4] = int(notetable3[k][4]) + int(y1e) - Extra_Edge_len
            notetable3[k][1] = int(notetable3[k][1]) + int(x2e)
            notetable3[k][3] = int(notetable3[k][3]) + int(x2e)
    if plotbar == 1:
        bar_code = (row_num) * 1000 + (bar_num)
        PLTshow( bar_code, section_onebar,1)
    return notetable3


def Add_SpiltCollect(notetable_Collect,spilt_Collect):
    NTC = [0,0,0,0,0,0,0,0]
    l_NTC = len(notetable_Collect)
    for i in range(l_NTC):
        NTC = np.vstack( (NTC, notetable_Collect[i,:]) )
        if i == l_NTC - 1 or notetable_Collect[i,7] != notetable_Collect[i+1,7]:
            n1 = notetable_Collect[i,6]
            n2 = notetable_Collect[i,7]
            SC_add = np.hstack( (spilt_Collect[int(n1) - 1,:],n1,n2) )
            NTC = np.vstack( (NTC, SC_add) )
    NTC = np.delete(NTC, 0, axis = 0)#删去第一行000
    return NTC


def Print_in_UI_1(self, start, Process_Title, p_title, p_time):
    try:
        if p_title == 1:
            self.ConsoleOutput_textbox.append(Process_Title)
        if p_time == 1:
            My_text = '[ ' + str(format(time.time() - start,'.4f')) + " seconds ]"
            self.ConsoleOutput_textbox.append(My_text)
        self.ConsoleOutput_textbox.append('')
        QtWidgets.QApplication.processEvents()
    except:
        print('No App running, pass it')


#%% Core！
#%%
def Core(self,Init_Img,po0,po1,path_ori):
    
    # print(path)
    # print('---------------------')
    path = self.OrigionImg_Path
    # slash_locate = str.rfind(path,'/')
    # Output_Address_FromfileName = path[ : int(slash_locate) ]
    # Img = cv2.imread(path)
    Img = Init_Img

    ImgSized3 = SizeImage(Img)#彩色图片
    
    
    ImgSized4 = cv2.GaussianBlur(ImgSized3,(7,7),3) #强化小节线
    Img01 = BinaryOperation3(ImgSized4, 215) 
    ImgSized3 = Enhance_BarLine(Img01, ImgSized3)
    
    self.ImgFin_Bin_0 = ImgSized3.copy()
    
    #%% 实验区域
    if Rotate_it == 1:
        Img = Rotate_Img0(ImgSized3,ImgSized3)
        Imshow_CV2('Img After Rotate', Img)

    start = time.time()
    if BinaryMethod == 0:
        ImgSized = BinaryOperation0(ImgSized3, 255) #选择传统二值化或自编
    else:
        ImgSized = BinaryOperation2(ImgSized3, 148) 
    Print_in_UI_1(self, start, 'Binary Finished', 1, 1)
   

    self.Image_CaseB = ImgSized.copy()#复制一份清理过的图像
    Image_CaseB_RGB = cv2.cvtColor(self.Image_CaseB, cv2.COLOR_GRAY2RGB)
    MYUI_NP_Plot(self,Image_CaseB_RGB)#Img After Del
     
    #self.ImgFin_Bin_1 = self.Image_CaseB
    self.ImgFin_Bin_0 = self.Image_CaseB.copy()
    
    ImgSized_LineRmoved,ImgSized_LineRmoved_01 = RemoveLines1(ImgSized)

    start = time.time()
    ImgSized_Del = del_SmallConnectedRegion(ImgSized, 360)#可容忍的最小连通域面积
    Print_in_UI_1(self, start, 'Cleaning Finished', 1, 1)
            

    
    ImgSized01 = np.where(ImgSized==0,1,0)

    # ImgSized_Del2 = ImgSized_Del.copy()
    # ImgSized_Del3 = cv2.cvtColor(ImgSized_Del2, cv2.COLOR_GRAY2RGB)
    # MYUI_NP_Plot(self,ImgSized_Del3) #Img After Del
    
    pixel_num = Caculate_RowPix(ImgSized_Del)#Plot中红色线2
    
    start = time.time()
    [r_distance,l_distance] = Caculate_DistanceFromSide(ImgSized_LineRmoved_01)
    Print_in_UI_1(self, start, 'Distance Caculating Finished', 1, 1)
    
    Plot_DivRowOutcome('1 Binaryed InitialImg',r_distance,pixel_num,l_distance,'0',path_ori)
    
    [Left_Start,Right_Start] = Caculate_ColPix(ImgSized_LineRmoved_01)#找到左侧乐谱开始的位置

    l_distance_DelMark = []
    r_distance_Memory_DelMark = []
    for i in range(len(l_distance)):#用于去除紧贴右下角的水印
       if l_distance[i] < max(l_distance):
           # if l_distance[i] > max(l_distance)*0.2:
           if l_distance[i] > Left_Start + max(l_distance)*0.1:
               l_distance_DelMark.append(i)#防止切割掉左侧无小节线的引子，part1
               r_distance_Memory_DelMark.append(r_distance[i])
               r_distance[i] = 0
               
    MrD = max(r_distance)
    n = 0
    for i in l_distance_DelMark:
        if r_distance_Memory_DelMark[n] > MrD - 12:#防止把原来不顶右边的杂物靠过来
            r_distance[i] = MrD#防止干扰最大值
        n += 1
    
    Plot_DivRowOutcome('2 After Del&RecoTooFarRowsFromLeft',r_distance,pixel_num,l_distance,'0',path_ori)
    
    [r_distance,l_distance,pixel_num] = Save_LeastRow(r_distance,l_distance,pixel_num,38,Right_Start)
    Plot_DivRowOutcome('3 After SavingLesatRow by larger38',r_distance,pixel_num,'0','0',path_ori)
    
    
    [r_distance,l_distance,pixel_num] = RemoveRows_DistinctivelyFarFromRight(r_distance,l_distance,pixel_num,1,10)
   
    DelMark = np.zeros(len(r_distance))
    for i in l_distance_DelMark:
        DelMark[i] = -88
        
    Plot_DivRowOutcome('4 After DelTooFarRowsFromRight',r_distance,pixel_num,l_distance,DelMark,path_ori)
    
    [r_distance,l_distance,pixel_num,y1,y2] = RemoveRows_ThinerThanThreshold(r_distance,l_distance,pixel_num,38)
    
    for i in l_distance_DelMark:#防止切割掉左侧无小节线的引子，part2
        l_distance[i] = 0
        r_distance[i] = 0
        pixel_num[i] = 0
    
    Plot_DivRowOutcome('5 After DelThinRaws by 38',r_distance,pixel_num,l_distance,DelMark,path_ori)

    [r_distance,l_distance,pixel_num] = RemoveRows_DistinctivelyFarFromRight(r_distance,l_distance,pixel_num,2,30)
    Plot_DivRowOutcome('6 After RemoveLowerRawsTwice',r_distance,pixel_num,'0','0',path_ori)
           
    [r_distance,l_distance,pixel_num,y1,y2] = RemoveRows_ThinerThanThreshold(r_distance,l_distance,pixel_num,26)
    Plot_DivRowOutcome('7 After DelThinRaws by 26',r_distance,pixel_num,'0','0',path_ori)

    [y1,y2,r_distance,pixel_num,avg_pixel_num] = Abbrivate_ByPixel(y1,y2,r_distance,pixel_num)
    Plot_DivRowOutcome('8 After AbbrivateBypixel_num',r_distance,pixel_num,'0','0',path_ori)

    [y1,y2] = RemoveRows_DetcetPossibleWriter(Right_Start,Left_Start,ImgSized01,y1,y2,avg_pixel_num)
    [r_distance,pixel_num] = Adjust_LineChartIndex(r_distance,pixel_num,y1,y2)
    Plot_DivRowOutcome('9 After Remove Writer',r_distance,pixel_num,'0','0',path_ori)
    
    [y1,y2] = RemoveRaws_ThinerThanAvg(y1,y2)
    [r_distance,pixel_num] = Adjust_LineChartIndex(r_distance,pixel_num,y1,y2)
    Plot_DivRowOutcome('10 After DelThinRaws by AVG',r_distance,pixel_num,'0','0',path_ori)
    
    Print_in_UI_1(self, start, '[' + str(len(y1)) + '] rows have been recongized.', 1, 0)
    
    
#%% 收集图片
    class X12:
        def __init__(self, x1, x2, Reco_bars):
            self.x1 = x1
            self.x2 = x2
            self.yes = Reco_bars#可以识别
    X12s = []
    
    Section_rows = []
    
 
    class ImgSection:
        def __init__(self, row_num, bar_num, img_bar, y1e, x2e, IsSaved):
            self.row = row_num
            self.bar = bar_num
            self.img = img_bar
            self.y1e = y1e
            self.x2e = x2e
            self.Saved = IsSaved        
    ImgSections = []
    

    spilt_Collect = []
    Extra_Edge_len = 36
    y_delete = []
        
    #Imshow_CV2('whoel2bbar', ImgSized)
    for i in range(len(y1)): #一行一行处理
        row_avaiavle = 1
        Reco_bars = 1
        x1,x2,spilt_Collect = SpiltWhole_ToBar(ImgSized,y1,y2,i,spilt_Collect,Left_Start) 
        if len(x1) == 1:
            Reco_bars = 0
            print('No bar detected, it might because raw div failed or s successfulkip a unrelevent row')
            if i == 0:
                print('The first line is unusual, stop programme')
                y1 = []#触发报错
                break
            
            
            #%% 循环救回
            print('start!')
            section_row =  ImgSized[y1[i] - 2: y2[i] + 2, :]
            Imshow_CV2('A questioned row',section_row)
            for cycle in range(1):
                notestr = get_text_Mixed(section_row,OCR_model)#不知道为什么原来是'num11'
                if len(notestr) == 0:
                    print('Warning! A bar seems be empty?:')
                else:
                    a_notetable = get_table(notestr, 0, 0)
                    
                    # section_row_show = ImgSized3[y1[i] - 2: y2[i] + 2, :]
                    #section_row_show = Draw_Reco(a_notetable,section_row_show)
                    #Imshow_CV2('sssss', section_row_show)
                    # section_row_show = 0
                    
                    avg_y1 = []
                    avg_y2 = []
                    for ii in range(len(a_notetable)):
                        RawNum = a_notetable[ii][0]
                        if str.isnumeric(RawNum) or RawNum == 'X':
                            y1p = int(a_notetable[ii][2])
                            y2p = int(a_notetable[ii][4])
                            Div = int( 4 / (cycle + 1))
                            avg_y1.append(y1p + y1p%Div)#化为偶数方便处理
                            avg_y2.append(y2p + y2p%Div)
                    i_max = max(np.bincount(avg_y1)) - 1 
                    y1p = avg_y1[ i_max ]#取出众数
                    i_max2 = max(np.bincount(avg_y2)) - 1 
                    y2p = avg_y1[ i_max2 ]#取出众数
                    y1[i] += y1p
                    if y2[i] - y1[i] < 30:#防止出现的干扰列
                        print('Double Save')
                        y1[i] -= y1p
                        avg_y11 = []
                        for iii in range(len(avg_y1)):
                            if avg_y1[iii] != y1p:
                                avg_y11.append(avg_y1[iii])
                        i_max = max(np.bincount(avg_y11)) - 1 
                        y1p = avg_y11[ i_max ]#取出众数
                        y1[i] += y1p
                    
                    
                    section_row = ImgSized[y1[i] - 2 : y2[i] + 2 , :]
                    Imshow_CV2('A row' + str(cycle),section_row)
                    
                    x1,x2,spilt_Collect = SpiltWhole_ToBar(ImgSized,y1,y2,i,spilt_Collect,Left_Start) 
            Reco_bars = 1
            print('Saved it')
            if len(x1) < 6:
                row_avaiavle = 0
                print('—————Passed it!——————')
                y_delete.append(i)
        
        if row_avaiavle == 1:
            X12_one = [x1,x2,Reco_bars]
            X12s.append(X12_one)
          
            
    y1 = np.delete(y1,y_delete)#删去可能是歌词的错误识别行
    y2 = np.delete(y2,y_delete)
            
    
    #%% Spilt and clean
    ImgSized3 = cv2.cvtColor(ImgSized3,cv2.COLOR_BGR2RGB)
    
    ImgSized31 = ImgSized3.copy()
    ImgSized31 = Draw_LocatedBars(y1,y2,X12s,ImgSized31)
    MYUI_NP_Plot(self,ImgSized31)
    
    
    self.ImgFin_Bin_1 = ImgSized31
    
    start = time.time()
    ress = []
    for i in range(len(y1)):
        ress.append( po0.apply_async(Spilt_and_Clean, (y1,y2,X12s,ImgSized,i,Extra_Edge_len) ) )
        
    po0.close()  # 关闭进程池， 关闭后po将不再接受新的请求
    po0.join()  # 等待po中所有子进程执行完成， 必须放在close语句后
    Print_in_UI_1(self, start, "Spilt finished", 1, 1)
    
    
    for i in range(len(y1)):
        BIDs,section_onerow = ress[i].get()
        Section_rows.append(section_onerow)
        for i2 in range(len(BIDs)):
            BID = BIDs[i2]
            B = ImgSection(BID[0], BID[1], BID[2], BID[3], BID[4], BID[5])
            ImgSections.append(B)
  
    h = []
    w = []
    for i in range(0,len(y1)):
        OutImg_sec = Section_rows[i]
        h.append(len(OutImg_sec))
        w.append(len(OutImg_sec[0]))
    W = max(w)
    H = sum(h)
    print(h)
    OutImg = np.uint8( np.zeros([H,W]) + 255 )
    
    for i in range(1,len(y1) + 1):
        OutImg_sec = np.uint8(Section_rows[i - 1])
        # hh = (len(OutImg_sec))
        # ww = (len(OutImg_sec[0]))
        # print(hh,ww)
        # print(i)
        OutImg[ (sum(h[0: i-1])) : sum(h[0: i]) , 0: w[i - 1] ] = OutImg_sec
        
    #Imshow_CV2('ssss', OutImg)
  
#%% OCR
    ISe = ImgSections
    notetable_Collect = []
    
    start = time.time()
    res = []
    if Multitask == 1:#Should be 1
        for n in range(len(ISe)):
            Se = ISe[n]
            res.append(po1.apply_async\
                          (TextReco_And_AddToNotetable, \
                          (Se.img,Se.y1e,Se.x2e,Extra_Edge_len,Se.row,Se.bar,OCR_model,plotbar)))
        po1.close()  # 关闭进程池， 关闭后po将不再接受新的请求
        po1.join()  # 等待po中所有子进程执行完成， 必须放在close语句后
        Print_in_UI_1(self, start, "Ocr finished", 1, 1)
        notetable_Collect = res[0].get()
        for i in range(1,len(res)):
            notetable_Collect = np.vstack((notetable_Collect, (res[i].get())))
    
    else:
        #这是不使用进程池的代码
        print("Start ocr")
        for n in range(len(ISe)):
            Se = ISe[n]
            notetable = TextReco_And_AddToNotetable(Se.img,Se.y1e,Se.x2e,Extra_Edge_len,Se.row,Se.bar,OCR_model,plotbar)
            if len(notetable_Collect) > 0:
                notetable_Collect = np.vstack( (notetable_Collect,notetable) )
            else:
                notetable_Collect = notetable
        print("Ocr finished")
      
    NTC = Add_SpiltCollect(notetable_Collect,spilt_Collect)
    
    #%% 去除异常识别结果
    #第一个是乱识别出来的数字
    Heights = []
    for i in range(len(NTC)):
        RawNum = str( NTC[i][0] )
        if str.isnumeric(RawNum) or RawNum == 'X':
            Heights.append(int(NTC[i][4]) - int(NTC[i][2]))
    Height_mean = np.mean(Heights)
    
    Widths = []
    for i in range(len(NTC)):
        RawNum = str( NTC[i][0] )
        if str.isnumeric(RawNum) or RawNum == 'X':
            Widths.append(int(NTC[i][4]) - int(NTC[i][2]))
    Width_mean = np.mean(Widths)
    
    for i in range(len(NTC)):
        RawNum = str( NTC[i][0] )
        if str.isnumeric(RawNum) or RawNum == 'X':
            if (int(NTC[i][4]) - int(NTC[i][2])) < Height_mean * 0.85:
                NTC[i][0] = '?'
            if (int(NTC[i][3]) - int(NTC[i][1])) > Width_mean * 1.5:
                NTC[i][0] = '?'
    
    #第二个是装饰音号下面多出的-（也可能是数字下面的）
    global NTC_temp
    NTC_temp = []
    for i in range(len(NTC)):
        dele_it = 0
        RawNum = str( NTC[i][0] )
        if RawNum == '-':
            pass
            try:
                if int(NTC[i - 1][1]) < int(NTC[i][3]) and int(NTC[i - 1][3]) > int(NTC[i][1]):
                    dele_it = 1
            except: pass
            try:
                if int(NTC[i + 1][1]) < int(NTC[i][3]) and int(NTC[i + 1][3]) > int(NTC[i][1]):
                    dele_it = 1
            except: pass
        

        
        if dele_it == 0:
            try:
                NTC_temp = np.vstack( (NTC_temp,NTC[i,:]))
            except:
                NTC_temp = NTC[i,:]
    NTC = NTC_temp
    
    
    #还原被识别为“-”的附点
    for i in range(len(NTC)):
        RawNum = str( NTC[i][0] )
        if RawNum == '-':
            if (int(NTC[i][4]) - int(NTC[i][2])) * 2 > (int(NTC[i][3])-  int(NTC[i][1])):
                NTC[i][0] = '.'

    
    #然后清除掉莫名部分
    NTC_temp = []
    for i in range(len(NTC)):
        if str(NTC[i][0]) != '?':
            try:
                NTC_temp = np.vstack( (NTC_temp,NTC[i,:]))
            except:
                NTC_temp = NTC[i,:]
    NTC = NTC_temp
    
    
    #清除掉没有紧跟数字的附点
    if str( NTC[1][0] ) == '.':
        NTC[1][0] = '?'
    for i in range(len(NTC)):
        RawNum = str( NTC[i][0] )
        if RawNum == '.':
            if str.isnumeric(NTC[i-1][0]):
                Distance_Between_Dot = int(NTC[i][1]) - int(NTC[i-1][3])
                #print(Distance_Between_Dot)
                if Distance_Between_Dot > 22:#如果隔得太远
                    NTC[i][0] = '?'
            else:
                NTC[i][0] = '?'
            
                
    #然后清除掉莫名部分
    NTC_temp = []
    for i in range(len(NTC)):
        if str(NTC[i][0]) != '?':
            try:
                NTC_temp = np.vstack( (NTC_temp,NTC[i,:]))
            except:
                NTC_temp = NTC[i,:]
    NTC = NTC_temp    
    
    
    Img_Recoed = Draw_Reco(NTC,ImgSized3)
    
    self.ImgFin_Recoed_0 = Img_Recoed

    return OutImg,Img_Recoed,NTC,path,y1

    
#%% Go to Lilypond Part
#%%
def BinaryOperation(image,
                    threshold: int):  
    Grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 先要转换为灰度图片
    ret, thresh = cv2.threshold(Grayimg, 150, 255,cv2.THRESH_BINARY) # 这里的第二个参数要调，是阈值
    return thresh


def Check_Dot(Fig, Direction, ii):
    DotLocation = []
    
    if Direction == 'down':
        pixel_compensate_y = 3
        pixel_compensate_x = 0
    elif Direction == 'up':
        pixel_compensate_y = 0
        pixel_compensate_x = 5#上面加防止识别入连音线
    else:
        pixel_compensate_y = 0
        pixel_compensate_x = 0
    image_array = 255 - Fig
    Octave_add = 0
    if sum(sum(image_array)) > 2550:#像素太少直接过
        background= 0
        labeled_image, num = measure.label(image_array, connectivity=2, background=background, return_num=True)#使用四联通，因为数字不会被误伤
        regions = measure.regionprops(labeled_image)
        region_added = []
        for x_half in range( pixel_compensate_y, math.floor(0.5* len(image_array)) - pixel_compensate_y):#防止括号乱入
            for y_half in range( pixel_compensate_x, math.floor(0.5 * len(image_array[0]))):#进行扫描，找到纵向贯穿的竖线分小节
                x = 2*x_half
                y = 2*y_half
                if labeled_image[x, y] != 0:
                    if labeled_image[x, y] in region_added:
                        pass
                    else:
                        region_added.append(labeled_image[x, y])
                        Lbound = regions[labeled_image[x, y] - 1].bbox[1]
                        Rbound = regions[labeled_image[x, y] - 1].bbox[3]
                        Ubound = regions[labeled_image[x, y] - 1].bbox[0]
                        Dbound = regions[labeled_image[x, y] - 1].bbox[2]
                        w = Rbound - Lbound
                        h = Dbound - Ubound
                        if 8<=w<=24 and 8<=h<=24 and abs(w - h) < 5 + round(min(h, w) / 6):
                            if regions[labeled_image[x, y] - 1].area > 0.6*w*h:
                                Octave_add += 1 
                                DotLocation_Add = [ii, Ubound, Dbound, Lbound, Rbound]
                                try:
                                    DotLocation = np.vstack([DotLocation, DotLocation_Add])
                                except:
                                    DotLocation = DotLocation_Add
    return Octave_add, DotLocation


def Check_UnderLine(i,ImgSized,NTC,LineSum_Memory,Pre_UnderLine_Height):
        Location_Adjust = 0
        LineLocation_Add = []
        
        if LineSum_Memory == 1:#上一个音符不存在下划线
            Scan_range = [int( NTC[i][1] ) + round(0.1 * (int( NTC[i][3] ) - int( NTC[i][1] ))),\
                          int( NTC[i][3] )]
        else:
            Scan_range = [int( NTC[i][1] ) + round(0.1 * (int( NTC[i][3] ) - int( NTC[i][1] ))),\
                          int( int( NTC[i][3] ) - round(0.25 * (int( NTC[i][3] ) - int( NTC[i][1] ))) )]
        LowerFig = ImgSized[int( NTC[i][4] ) - 2 : int( NTC[i][4] ) + 40, Scan_range[0]: Scan_range[1]]
        
        
        End_UnderLine_ii = 4#用于检测点的时候跳过一小段
        
        LFsum = np.sum(LowerFig, axis=1)#*255，the more black, the smalle, even 0
        LFstart = 0
        LineSum = 1#一根下划线后0.5——1/8，两根后0.25——1/16
        
        
        UnderLine_Height = 0
        ScanR_length = round(0.45 * 255 * (Scan_range[1] - Scan_range[0]));
        for ii in range(0, len(LFsum)-1):
            if int(LFsum[ii]) - int(LFsum[ii + 1]) > ScanR_length and LFstart == 0:
            #if LFsum[ii] > 4 * LFsum[ii+1]:
                LFstart = 1
                ii_LFstart = ii
                continue;
            if LFstart == 1:
                if LFsum[ii] * 1.2 > LFsum[int(ii_LFstart)] or LFsum[ii] > ScanR_length * 1.2:#防止黏连下划线的点延厚下划线
                    LFstart = 0
                    LineSum = LineSum/2
                    End_UnderLine_ii = ii + 2 #来自于LowerFig的选取方案
                    UnderLine_Height = UnderLine_Height + (ii - ii_LFstart)
                    LineLocation_Add_Pre = [i, int( NTC[i][4] ) - 2 + ii_LFstart, int( NTC[i][4] ) - 2 + ii, Scan_range[0], Scan_range[1]]
                    try:
                        LineLocation_Add = np.vstack([LineLocation_Add, LineLocation_Add_Pre])
                    except:
                        LineLocation_Add = LineLocation_Add_Pre
        
        if Pre_UnderLine_Height != 0:
            if LineSum_Memory == 0.5:
                if LineSum == 0.5 and Pre_UnderLine_Height + 1 < UnderLine_Height * 0.75:
                    LineSum = LineSum/2
                    Location_Adjust = 1
            if LineSum_Memory == 0.25:
                if LineSum == 0.5 and Pre_UnderLine_Height + 1 < UnderLine_Height * 1.25:
                    LineSum = LineSum/2
                    Location_Adjust = 1
            
        if  Location_Adjust == 1:
            if (LineLocation_Add[0,3] - LineLocation_Add[0,2]) > (LineLocation_Add[1,3] - LineLocation_Add[1,2]):
                pass#Spilt 0 to 2 parts
            else:
                pass
                
            
        # if i < 60:
        #     PLTshow(str(i) + '_' + str(UnderLine_Height) + "+" + str(LineSum),LowerFig,1)
        #     print("\n____")
        #     print(i)
        #     print(ScanR_length)
        #     print(LFsum)
             
        LineSum_Memory = LineSum
        return LineSum, LineSum_Memory, End_UnderLine_ii, UnderLine_Height, LineLocation_Add
    
    
def TuneMapping(Tune, Tune_Adjustment):
    
    Octave_Base = 5
      
    if Tune == 'c':
        Tune_Adjust = [0,0,0,0,0,0,0,0]
    elif Tune == 'g':
        Tune_Adjust = [0,0,0,0,1,0,0,0]
    elif Tune == 'd':
        Tune_Adjust = [0,1,0,0,1,0,0,0]
    elif Tune == 'a':
        Tune_Adjust = [0,1,0,0,1,1,0,0]
    elif Tune == 'e':
        Tune_Adjust = [0,1,1,0,1,1,0,0]
    elif Tune == 'b':
        Tune_Adjust = [0,1,1,0,1,1,1,0]
        
    elif Tune == 'f':
        Tune_Adjust = [0,0,0,0,0,0,0,-1]
    elif Tune == 'bes':
        Tune_Adjust = [0,0,0,-1,0,0,0,-1]
    elif Tune == 'ees':
        Tune_Adjust = [0,0,0,-1,0,0,+11,-1]
    elif Tune == 'aes':
        Tune_Adjust = [0,0,-1,-1,0,0,+11,-1]
    elif Tune == 'des':
        Tune_Adjust = [0,0,-1,-1,0,-1,+11,-1]
    elif Tune == 'ges':
        Tune_Adjust = [0,-1,-1,-1,0,-1,+11,-1]
        
    if Tune_Adjustment != 0:
        TuneCycle = ['c', 'b', 'd', 'ees', 'e', 'f', 'ges', 'g', 'aes', 'a', 'bes', 'b']
        NewLocate_TuneCycle = (TuneCycle.index(Tune) + Tune_Adjustment) % 12
        Tune = TuneCycle[NewLocate_TuneCycle]
        Octave_Base = Octave_Base + int(np.fix(Tune_Adjustment / 12))
        
    Trans_CM = [0,4,6,8,9,11,1,3]#1对应C,0对应0
    Trans_CM = np.sum( [Trans_CM, Tune_Adjust], axis = 0)#调整音高

    if min(Tune_Adjust) >= 0:#决定用升号或降号
        NumToNote = ['r','a','ais','b','c','cis','d','dis','e','f','fis','g','gis']#第一个应为0
    else:
        NumToNote = ['r','a','bes','b','c','des','d','ees','e','f','ges','g','aes']#第一个应为0
    
    print('The Adjusted Tune is:')
    print(Tune)
    
    return Tune, Trans_CM, NumToNote, Octave_Base


#%% Core2!
#%%
def Core2(self,NTC):
    class NTCC2:
        def __init__(self, note, octave, coordinate, row_num, bar_num, duration, IsNum):
            self.note = note
            self.oct = octave
            self.coord = coordinate
            self.row = row_num
            self.bar = bar_num
            self.dur = duration
            self.IsN = IsNum
    
    Spilt_Method = 1 #0为5个一换，1为一行一换
    
    OCR_model = 'num13.1'
    # Img3 = cv2.imread(self.OrigionImg_Path)
    # ImgSized3 = SizeImage(Img3)#彩色图片
    # ImgSized = BinaryOperation(ImgSized3, 240) #二值化，黑白阈值（黑为255）
    
    
    notetableCollect = NTC
    
    print('\nDocument inpiut:')
    print(notetableCollect)
    print('')
    NTC = notetableCollect
    
    (h,w) = self.Image_CaseB.shape
    LSCollect = []#0为无关，1为整拍，2为二分之一
    ISNum = []#1是,0否
    Octaves = []#A4
    
    LineSum_Memory = 1
    UnderLine_Height = 0
    
    Tune = str.lower( self.Tune_textbox.toPlainText() )#引入调性与映射
    Tune_Adjustment = int( self.Tune_Adjust_textbox.toPlainText() )
    Tune, Trans_CM, NumToNote, Octave_Base = TuneMapping(Tune, Tune_Adjustment)
    
    LineLocation = [0,0,0,0,0]#序号，X12s,y1,y2
    DotLocation_U = [0,0,0,0,0]
    DotLocation_D = [0,0,0,0,0]
    
    for i in range(0,len(NTC)):
        i_ori = i
        
        RawNum = str( NTC[i][0] )
        oct_added = 0
        
        if RawNum == '|':
            LineSum_Memory = 1#换小节时重置前一段的记录

        if str.isnumeric(RawNum) or RawNum == 'X':#如果是数字
            try:
                RawNum  = int(RawNum)
            except:
                RawNum = 1#记得改，用于X的时候
                
            ISNum.append(1)
            #%% 检测下划线
            LineSum, LineSum_Memory, End_UnderLine_ii, UnderLine_Height, LineLocation_Add = \
                Check_UnderLine(i,self.Image_CaseB,NTC,LineSum_Memory,UnderLine_Height)
            LSCollect.append(LineSum)
            
            if len(LineLocation_Add) != 0:
                LineLocation = np.vstack([LineLocation, LineLocation_Add])
            # Fig_Accurate = ImgSized[int( NTC[i][2] ): int( NTC[i][4] ), \
            #                         int( NTC[i][1] ): int( NTC[i][3] )]
            # section = Fig_Accurate
            #%% 检测点
            UpperFig = self.Image_CaseB[int( NTC[i][2] ) - 32 - 10: int( NTC[i][2] ) - 6, \
                                int( NTC[i][1] ) - 10: int( NTC[i][3] ) + 10]
            # PLTshow('ok',UpperFig,1)
            Octave_add_u, DotLocation_U_Add = Check_Dot(UpperFig, 'up', i)
            if DotLocation_U_Add != []:
                DotLocation_U_Add = np.array( DotLocation_U_Add )
                try:
                    DotLocation_U_Add[:,1] -= (int( NTC[i][2] ) - 6 - 5)#因为有5个像素的总想保护
                    DotLocation_U_Add[:,2] -= (int( NTC[i][2] ) - 6 - 5)
                    DotLocation_U_Add[:,3] += (int( NTC[i][1] ) - 10)
                    DotLocation_U_Add[:,4] += (int( NTC[i][1] ) - 10)
                except:
                    Temp_Location1 = DotLocation_U_Add[1]
                    DotLocation_U_Add[1] = (int( NTC[i][2] ) - 6 ) -  DotLocation_U_Add[2]
                    DotLocation_U_Add[2] = (int( NTC[i][2] ) - 6 ) -  Temp_Location1 
                    DotLocation_U_Add[3] += (int( NTC[i][1] ) - 10)
                    DotLocation_U_Add[4] += (int( NTC[i][1] ) - 10)
                
                DotLocation_U = np.vstack([DotLocation_U, DotLocation_U_Add])
            
            
            LowerFig = self.Image_CaseB[int( NTC[i][4] ) + End_UnderLine_ii: int( NTC[i][4] ) + End_UnderLine_ii + 30, \
                                int( NTC[i][1] ) - 3: int( NTC[i][3] ) + 3] #检测下方时，左右加一点，防止下划线被切割得像是点
            Octave_add_d, DotLocation_D_Add = Check_Dot(LowerFig, 'down', i)
            if DotLocation_D_Add != []:
                DotLocation_D_Add = np.array( DotLocation_D_Add )
                try:
                    DotLocation_D_Add[:,1] += (int( NTC[i][4] ) + End_UnderLine_ii)
                    DotLocation_D_Add[:,2] += (int( NTC[i][4] ) + End_UnderLine_ii)
                    DotLocation_D_Add[:,3] += (int( NTC[i][3] ) - 3 + 3)#因为有3个像素的横向保护
                    DotLocation_D_Add[:,4] += (int( NTC[i][3] ) - 3 + 3)
                except:
                    DotLocation_D_Add[1] += (int( NTC[i][4] ) + End_UnderLine_ii)
                    DotLocation_D_Add[2] += (int( NTC[i][4] ) + End_UnderLine_ii)
                    DotLocation_D_Add[3] += (int( NTC[i][1] ) - 3 )
                    DotLocation_D_Add[4] += (int( NTC[i][1] ) - 3 )
                
                DotLocation_D = np.vstack([DotLocation_D, DotLocation_D_Add])
            
            # if i < 330:
            #     PLTshow(str(i) + '_' + str(Octave_add_u), UpperFig,1)
       
            Octaves.append(Octave_add_u - Octave_add_d + Octave_Base)#4是基本音高,补偿1,后面装饰音还有
            oct_added = 1
            
            


        #%% 数字以外情况
        elif RawNum == '-':
            LSCollect.append(1)
            ISNum.append(0)
        elif RawNum == '.':
            LSCollect.append( (LSCollect[-1] * 0.5) )
            ISNum.append(0)
            
            
        elif RawNum == '^':
            LSCollect.append(0)
            ISNum.append(0)
            UpperFig = self.Image_CaseB[int( NTC[i_ori][2] ) - 80: int( NTC[i_ori][2] ) - 3, \
                                int( NTC[i_ori][1] ) - 20: int( NTC[i_ori][3] ) + 20]
            (h,w) = UpperFig.shape
            UpperFig = cv2.resize(UpperFig,(w*2,h*2))
            #Imshow_CV2('ssssss',UpperFig)
            
            print('Searching figure for ^')
            notestr = get_text(UpperFig,OCR_model)
            text = get_table(notestr,0,0)
            Nums_i = []
            Deco_note = []
            try:
                np.size(text,1)#防止text矩阵只有一行，导致报错
                for ii in range(len(text)):
                    RawNum = str(text[ii][0])
                    if str.isnumeric(RawNum) :
                        Nums_i.append(ii)
            except:
                RawNum = str(text[0][0])
                if str.isnumeric(RawNum) :
                    Nums_i.append(0)
                    
            if len(Nums_i) == 0:
                print('Opps,no decorate note can be found!')
            elif len(Nums_i) == 1:
                Deco_note = text[Nums_i][0][0]
                print('Deco it for: ',end = '')
                print(Deco_note)
                NTC[i_ori][0] = '^' + str(Deco_note)
            else:
                RawNum_Score = 0
                for j in range(len(Nums_i)):
                    index = text[Nums_i[j]]
                    New_RawNum_Score = (int(index[3]) - int(index[1]) + 30) * (int(index[4]) - int(index[2]))
                    if New_RawNum_Score > RawNum_Score:
                        Deco_note = index[0]
                        RawNum_Score = New_RawNum_Score
                print('Deco it for: ',end = '')    
                print(Deco_note)
                NTC[i_ori][0] = '^' + str(Deco_note)
            # print(text)
                
            if len(Deco_note) != 0:
                # Sub_UpperFig = ImgSized[int( NTC[i_ori][2] ) - 80: int( NTC[i_ori][2] ) - 3, \
                #                     int( NTC[i_ori][1] ) - 20: int( NTC[i_ori][3] ) + 20]
                # # PLTshow('ok',UpperFig,1)
                # Octave_add_u = Check_Dot(Sub_UpperFig)
                Octave_add_u = 0
                
                Sub_LowerFig = self.Image_CaseB[int( NTC[i_ori][2] ) - 32 - 10: int( NTC[i_ori][4] ) - 8, \
                                    int( NTC[i_ori][1] ) - 15: int( NTC[i_ori][3] )]
                (h,w) = Sub_LowerFig.shape
                Sub_LowerFig  = cv2.resize(Sub_LowerFig ,(round(w*1.5),round(h*1.5)))
                Octave_add_d, Trivial = Check_Dot(Sub_LowerFig, 'up', i)
                print(Octave_add_d)
                #Imshow_CV2('ssssss',Sub_LowerFig)
                PLTshow('ok',Sub_LowerFig,1)
                
                Octaves.append(Octave_add_u - Octave_add_d + Octave_Base)#5是基本音高
                oct_added = 1
            
        else:
            LSCollect.append(0)
            ISNum.append(0)
            
        if oct_added == 0:
            Octaves.append(0)
 
    
    #%% 整合
    LSCollect = np.array(LSCollect)
    ISNum = np.array(ISNum)
    
    global aaaNTC2
    aaaNTC2 = np.transpose( np.vstack( (NTC[:,0], Octaves, LSCollect, ISNum, NTC[:,6], NTC[:,7]) ) )
    
    NTC22 = []
    for i in range(len(NTC)):
        note = NTCC2(NTC[i,0], Octaves[i], NTC[i,1:5], NTC[i,6], NTC[i,7], LSCollect[i], ISNum[i])
        NTC22.append(note)
    
    
    #%% 节奏检查
    global NTC2_Check
    NTC2_Check = []
    Bar_Len = float(0)
    Bar_No = 1
    Bar_Lens = []
    Bar_Nos = []
    Row_Nos = []
    
    for i in range(0,len(NTC22)-1):
        if NTC22[i].bar == NTC22[i+1].bar:
            Bar_Len = Bar_Len + float(NTC22[i].dur)
        else:
            Bar_Lens.append(Bar_Len)
            Bar_Len = float(0)
            Bar_Nos.append(Bar_No)
            Bar_No = Bar_No + 1
            Row_No = int(NTC22[i].row)
            Row_Nos.append(Row_No)
    
    NTC2_Check = np.transpose( np.vstack( (Bar_Lens, Row_Nos, Bar_Nos) ) )
    BeatTime = int(round(np.mean(NTC2_Check[:,0])))
    
    #%% 画出识别效果图
    ImgSized32 = self.ImgSized32
    global Locations
    Dot_Locations = np.vstack([DotLocation_U, DotLocation_D])
    layer = np.zeros(ImgSized32.shape, np.uint8)
    layer = Creat_Layer(LineLocation,ImgSized32, [120,200,80], layer)
    layer = Creat_Layer(Dot_Locations,ImgSized32, [100,220,220], layer)
    Img_Recoed32 = cv2.addWeighted(ImgSized32, 0.7, layer, 0.95, 1)
    MYUI_NP_Plot(self,Img_Recoed32)
        
    self.ImgFin_Recoed_1 = Img_Recoed32

    
    #%% for的准备
    content = str()
    print_num = 0
    Pass = 0
    Div = 0
    
    #%% for循环，遍历每一个note
    for i in range(len(NTC22)-1):
        if Spilt_Method == 0:
            if int(NTC22[i].bar)%5 == 1 or i == (len(NTC22) - 1):
                if Div == 1:
                    Div = 0
                    print_num += 1
                    if print_num == 1:
                        content_Collect = content
                    else:
                        content_Collect = np.vstack( (content_Collect,content) )
                    content = str()
            else:
                Div = 1
                    
        if Spilt_Method == 1:
            if i > 0:
                if NTC22[i].row > NTC22[i-1].row or i == (len(NTC22) - 1):
                    print_num += 1
                    if print_num == 1:
                        content_Collect = content
                    else:
                        content_Collect = np.vstack( (content_Collect,content))
                    content = str()
            
        RawNum = str( NTC[i][0] )
        RawNum_i = i
        length = 4
        
        if Pass > 0:
            Pass = Pass - 1
            continue
       
        if str.isnumeric(RawNum) or RawNum == 'X':
            if RawNum == 'X':
                continue
            if RawNum == '0':
                pitch = int(RawNum)
                pitch = Trans_CM[pitch]
                pitch = NumToNote[pitch]
            if str.isnumeric(RawNum) and RawNum != '0':
                           
                pitch = int(RawNum)
                pitch_raw0 = Trans_CM[pitch]
                
                pitch_raw = pitch_raw0 + int(Tune_Adjustment - 12*np.fix(Tune_Adjustment/12))
                pitch = int( pitch_raw%12 )
                if pitch == 0:
                    pitch = 12
                if Tune_Adjustment >= 0:
                    octave_append_num = NTC22[i].oct - 3# + int(np.fix( (pitch_raw - 4) /12))
                    if (pitch_raw0 <= 3 and pitch_raw >= 4) or (pitch_raw0 > 3 and pitch_raw >= 16):
                        octave_append_num += 1 
                else:
                    octave_append_num = NTC22[i].oct - 3 
                    if (pitch_raw0 >= 4 and pitch_raw <= 3) or (pitch_raw0 < 4 and pitch_raw <= -9):
                        octave_append_num -= 1
                
                pitch = NumToNote[pitch]
                if octave_append_num == 0:
                    pitch = pitch
                elif octave_append_num > 0:
                    for i in range(octave_append_num):
                        pitch = pitch + str('\'')#这是'符号
                else:
                    for i in range(abs(octave_append_num)):
                        pitch = pitch + str(',')
                
            length = int(length*(1/float((NTC22[RawNum_i].dur))))
            
            ii = 1
            while str( NTC[RawNum_i + ii][0] ) == '-':
                Pass = Pass + 1
                ii = ii + 1
                if length == 4 or length == 8:#8必然是识别错误
                    length = 2
                elif length == 2:
                    length = '2.'
                elif length == '2.':
                    length = 1
                if i + ii == len(NTC22):#全曲结尾处及时停止
                    break
 
            #开始检测附点
            ii = 1
            while str( NTC[RawNum_i+ii][0] ) == '.':
                Pass = Pass + 1
                ii = ii + 1
                if length == 4:
                    length = '4.'
                elif length == 8:
                    length = '8.'
                else:
                    print('That dot(.) seems strange:',end = '')
                    print( NTC[RawNum_i+ii][6] , NTC[RawNum_i+ii][7] , [RawNum_i+ii] )
                    print('')
                    pass
                if i + ii == len(NTC22):#全曲结尾处及时停止
                    break
   
            content += str(pitch)
            content += str(length)
            content += ' '
        
        if RawNum[0] == '^':
            
            pitch = int(RawNum[1:])
            pitch_raw0 = Trans_CM[pitch]
            
            pitch_raw = pitch_raw0 + int(Tune_Adjustment - 12*np.fix(Tune_Adjustment/12))
            pitch = int( pitch_raw%12 )
            if pitch == 0:
                pitch = 12
            if Tune_Adjustment >= 0:
                octave_append_num = NTC22[i].oct - 3# + int(np.fix( (pitch_raw - 4) /12))
                if (pitch_raw0 <= 3 and pitch_raw >= 4) or (pitch_raw0 > 3 and pitch_raw >= 16):
                    octave_append_num += 1 
            else:
                octave_append_num = NTC22[i].oct - 3 
                if (pitch_raw0 >= 4 and pitch_raw <= 3) or (pitch_raw0 < 4 and pitch_raw <= -9):
                    octave_append_num -= 1
            
            pitch = NumToNote[pitch]
            if octave_append_num == 0:
                pitch = pitch
            elif octave_append_num > 0:
                for i in range(octave_append_num):
                    pitch = pitch + str('\'')#这是'符号
            else:
                for i in range(abs(octave_append_num)):
                    pitch = pitch + str(',')
                    
            
            content += (r'\appoggiatura {'  + pitch + '16}')
        
        Direct_content = ['|']
        if RawNum in Direct_content:
            content += (RawNum + ' ')
            
        if RawNum == '(':
            content += ('}My_Change_Row{ \tiny \time ' + str(BeatTime) + '/4' + r' \key ' + str(Tune) + r' \major ')

        if RawNum == ')':
            content += ('}My_Change_Row{ \time ' + str(BeatTime) + '/4' + r' \key ' + str(Tune) + r' \major ')

    content_Collect = np.vstack( (content_Collect,content))#添上最后一段
    return content_Collect, Spilt_Method, BeatTime, Tune
    
#%% Core3 Saving
def Core3(self, content_Collect, Spilt_Method, BeatTime, Tune):
    content_Collect_temp = []
    for i in range(len(content_Collect)):
        try:
            content_Collect_temp = np.vstack((content_Collect_temp, content_Collect[i]))
        except:
            content_Collect_temp = content_Collect[i]
        if Spilt_Method == 0:
            content_Collect_temp = np.vstack((content_Collect_temp, str('%-----{ Bar ' + str((i+1)*5) + ' } -----')) )
        else:
            content_Collect_temp = np.vstack((content_Collect_temp, str('%-----{ Row ' + str(i+1) + ' } -----')) )
        content_Collect_temp = np.vstack((content_Collect_temp,''))
    content_Collect= content_Collect_temp
    
    for j_half in range( int(len(content_Collect)/3) ):
        j = 3*j_half
        print(content_Collect[j])
        if Spilt_Method == 0:
            if j < len(content_Collect) - 1:
                print('-------------{ Bar ' + str((j_half+1)*5) + ' }-------------\n')
            else:
                print('-------------{ Bar ' + str(NTC[-1][7]) + ' }-------------\n')
        else:
            print('-------------{ Row ' + str((j_half+1)) + ' }-------------\n')
    
    self.Output_Filename = self.filename_textbox.toPlainText()   
            
    Title1 = 'version "2.22.2"  \
            \n\header{\
            \ntitle = "' + self.Output_Filename + '"\
            \nsubtitle = ""\n}\
            \n'
    Title2 = r'{'
    Title3 = '\n'
    Title4 = r'\time ' + str(BeatTime) + '/4 ' + r'\key ' + str(Tune) + ' \major \n'
    
    
    global C
    Title = Title1 + Title2 + Title3 + Title4
    self.Output_Address = self.OutputAddress_textbox.toPlainText()
    Note=open(self.Output_Address + '/Content_Test.txt',mode='w')
    C = str(content_Collect)
    C = C.replace('[\'', '')
    C = C.replace('\']', '')
    C = C.replace('[', '')
    C = C.replace(']', '')
    C = C.replace('\"', '')
    C = C.replace('\\a', 'a')#取消多余的\\
    C = C.replace('\\key', 'key')#取消多余的\\
    C = C.replace('\\major', 'major \n')#取消多余的\\，并加入\n
    C = C.replace('My_Change_Row', '\n')#为)(--}{加入\n
    C = Title + C + '\n}'
    Note.write(C)
    Note.close()
    
    filename = self.Output_Address + '/Content_Test.txt'
    portion = os.path.splitext(filename)  # 分离文件名与扩展名
    ly_i = 1
    ly_created = 0
    while ly_created == 0:
        try:
            newname = portion[0] + str(ly_i) + '.ly'
            os.rename(filename, newname)
            ly_created = 1
        except:
            ly_i += 1
    self.ly_path = newname
    
    
#%% main
if __name__ == '__main__':    

    plotbar = 0
    savedata = 1
    Rotate_it = 0
    Preview_outcome = 0
    Multitask = 1
    BinaryMethod = 1#1是新的
    Name_No = 'WithPsm13'
    OCR_model = 'num13.1'
    
    app = QApplication(sys.argv)
    ex = MyUI()
    ex.setWindowTitle("NMN Reco V2.1")
    ex.show()
    sys.exit(app.exec_())
        
        