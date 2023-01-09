import os
import sys
import time

from etc_QT import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import cv2
import numpy as np

# HSV_Target Database -- 推荐参数
lower_red2 = np.array([156, 80, 20])
higher_red2 = np.array([180, 255, 255])
lower_red = np.array([0, 160, 20])
higher_red = np.array([15, 255, 255])  # 15
lower_green = np.array([48, 65, 10])  # 55
higher_green = np.array([90, 255, 255])  # 90
lower_blue = np.array([100, 60, 25])  # 100
higher_blue = np.array([124, 255, 255])

# 阈值参数区
area_threshold = 1000           # 面积过滤阈值
# line_differenceValue = 20       # 线性关系偏差
# layer_highDec


class CV_thread(QThread):
    def __init__(self, parent=None):
        super(CV_thread, self).__init__(parent)
        self.Rpos = None
        self.Gpos = None
        self.Bpos = None

    def work(self):
        while self.isOpened:
            flag, frame = self.cap.read()
            frame_height, frame_width, _ = frame.shape

            # 层数分割
            # for i in range(0, frame_height / 2):
            #     for j in range(0, frame_width):
            #         frame1 = frame[i, j]
            # for i in range(frame_height / 2, frame_height):
            #     for j in range(0, frame_width):
            #         frame2 = frame[i, j]
            # while 1:
            #     cv2.imshow('frame1', frame1)
            #     cv2.imshow('frame2', frame2)
            #     cv2.waitKey(0)

            # frame = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame = cv2.erode(hsv_frame, None, iterations=2)

            # print(lower_red, higher_red, lower_green, higher_green, lower_blue, higher_blue)

            mask_red1 = cv2.inRange(hsv_frame, lower_red, higher_red)
            mask_red2 = cv2.inRange(hsv_frame, lower_red2, higher_red2)
            mask_red = mask_red1 + mask_red2
            mask_green = cv2.inRange(hsv_frame, lower_green, higher_green)
            mask_blue = cv2.inRange(hsv_frame, lower_blue, higher_blue)

            mask_red = cv2.erode(mask_red, None, iterations=2)
            mask_green = cv2.erode(mask_green, None, iterations=2)
            mask_blue = cv2.erode(mask_blue, None, iterations=2)
            mask_red = cv2.dilate(mask_red, (4, 4), iterations=5)
            mask_green = cv2.dilate(mask_green, (4, 4), iterations=5)
            mask_blue = cv2.dilate(mask_blue, (4, 4), iterations=5)

            # mask_green = cv2.GaussianBlur(mask_green, (3, 3), 0)
            # mask_red = cv2.GaussianBlur(mask_red, (3, 3), 0)
            # mask_blue = cv2.GaussianBlur(mask_blue, (3, 3), 0)

            mask_green = cv2.medianBlur(mask_green, 7)
            mask_red = cv2.medianBlur(mask_red, 7)
            mask_blue = cv2.medianBlur(mask_blue, 7)

            # line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, line)
            # mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, line)
            # mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, line)

            cnts1, contours1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts2, contours2 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts3, contours3 = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.circle(frame, (int(frame_width / 2), int(frame_height / 2)), 5, (255, 0, 0), 3)
            cntR = cntG = cntB = 0

            # 红色
            for cnt in cnts1:
                (x, y, w, h) = cv2.boundingRect(cnt)
                if w * h > area_threshold:
                    cntR += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Red Target,Pos:{(2 * x + w) / 2, (2 * y + h) / 2}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.Rpos = [1, (2 * x + w) / 2, (2 * y + h) / 2]

            # 绿色
            for cnt in cnts2:
                (x, y, w, h) = cv2.boundingRect(cnt)
                if w * h > area_threshold:
                    cntG += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Green Target,Pos:{(2 * x + w) / 2, (2 * y + h) / 2}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.Gpos = [2, (2 * x + w) / 2, (2 * y + h) / 2]

            # 蓝色
            for cnt in cnts3:
                (x, y, w, h) = cv2.boundingRect(cnt)
                if w * h > area_threshold:
                    cntB += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Blue Target,Pos:{(2 * x + w) / 2, (2 * y + h) / 2}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    self.Bpos = [3, (2 * x + w) / 2, (2 * y + h) / 2]

            # print(aim_color[0])
            print(f"{cntR} {cntG} {cntB}")

            if flag:
                cv2.imshow("Test", frame)
                cv2.imshow("red", mask_red)
                cv2.imshow("green", mask_green)
                cv2.imshow("blue", mask_blue)
                print(self.Rpos, self.Gpos, self.Bpos)
                posSort = [self.Rpos, self.Gpos, self.Bpos]

                if cntR == 1 and cntG == 1 and cntB == 1:
                    for i in range(0, 3):
                        for j in range(i + 1, 3):
                            if posSort[i][2] < posSort[j][2]:
                                t = posSort[i]
                                posSort[i] = posSort[j]
                                posSort[j] = t
                    for i in range(0, 3):
                        print(posSort[i][0])

            if cv2.waitKey(1) == ord("q"):
                break

    # def fun_lineMath(self, x1, y1, x2, y2):
    #     """线性关系运算"""
    #     # 线性判断算法    y = kx+b
    #     k = (y2 - y1) / (x2 - x1)
    #     b = y1 - k * x1
    #     print(f"y = {k}x + {b}")
    #     return k, b

    # def scanLayer(self):
    #     """层数判断算法"""
    #
    #     pass

    def destroyCV(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.isOpened = self.cap.isOpened()
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.work()
        self.destroyCV()


class mainWin(QMainWindow, Ui_ETC_UI_main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowIcon(QIcon('cy2.ico'))
        self.evt_run = CV_thread()
        self.textBrowser.setText(" <h1>我是小帮手绫乃,"
                                 " 这里的各项阈值给了默认值, 可开启测试进行动态调整</h1>")

        # 主程序按钮
        self.button_R.clicked.connect(self.go_R_page)
        self.button_G.clicked.connect(self.go_G_page)
        self.button_B.clicked.connect(self.go_B_page)
        self.button_run.clicked.connect(self.go_CV_thread)
        self.button_close.clicked.connect(self.evt_close)
        self.button_small.clicked.connect(self.evt_small)

        # 页面按钮
        self.Red_down.clicked.connect(self.go_Red_down_Page)
        self.Red_up.clicked.connect(self.go_Red_up_Page)
        self.Green_down.clicked.connect(self.go_Green_down_Page)
        self.Green_up.clicked.connect(self.go_Green_up_Page)
        self.Blue_down.clicked.connect(self.go_Blue_down_Page)
        self.Blue_up.clicked.connect(self.go_Blue_up_Page)

        # 色块动态检测
        self.Red_down_H.valueChanged.connect(self.colActDetect)
        self.Red_up_H.valueChanged.connect(self.colActDetect)
        self.Green_down_H.valueChanged.connect(self.colActDetect)
        self.Green_up_H.valueChanged.connect(self.colActDetect)
        self.Blue_down_H.valueChanged.connect(self.colActDetect)
        self.Blue_up_H.valueChanged.connect(self.colActDetect)

        self.Red_down_S.valueChanged.connect(self.colActDetect)
        self.Red_up_S.valueChanged.connect(self.colActDetect)
        self.Green_down_S.valueChanged.connect(self.colActDetect)
        self.Green_up_S.valueChanged.connect(self.colActDetect)
        self.Blue_down_S.valueChanged.connect(self.colActDetect)
        self.Blue_up_S.valueChanged.connect(self.colActDetect)

        self.Red_down_V.valueChanged.connect(self.colActDetect)
        self.Red_up_V.valueChanged.connect(self.colActDetect)
        self.Green_down_V.valueChanged.connect(self.colActDetect)
        self.Green_up_V.valueChanged.connect(self.colActDetect)
        self.Blue_down_V.valueChanged.connect(self.colActDetect)
        self.Blue_up_V.valueChanged.connect(self.colActDetect)
        # 窗口可拖动
        self.mouse_x = self.mouse_y = self.origin_x = self.origin_y = None

    # 1.鼠标点击事件
    def mousePressEvent(self, evt):
        # 获取鼠标当前的坐标
        self.mouse_x = evt.globalX()
        self.mouse_y = evt.globalY()

        # 获取窗体当前坐标
        self.origin_x = self.x()
        self.origin_y = self.y()

    # 2.鼠标移动事件
    def mouseMoveEvent(self, evt):
        # 计算鼠标移动的x，y位移
        move_x = evt.globalX() - self.mouse_x
        move_y = evt.globalY() - self.mouse_y

        # 计算窗体更新后的坐标：更新后的坐标 = 原本的坐标 + 鼠标的位移
        dest_x = self.origin_x + move_x
        dest_y = self.origin_y + move_y

        # 移动窗体
        self.move(dest_x, dest_y)

    # def layerScan(scanWidth):
    #     for i in range(0, 280, 20):
    #         time.sleep(0.5)
    #         frame1 = frame[i:i + scanWidth, 0:frame_width]
    #         frame2 = frame[480 - i - scanWidth:480 - i, 0:frame_width]
    #         cv2.imshow('frame1', frame1)
    #         cv2.imshow('frame2', frame2)
    #         if cv2.waitKey(1) == ord("q"):
    #             break

    def colActDetect(self):
        global lower_red, higher_red, lower_red2, higher_red2, lower_green, higher_green, lower_blue, higher_blue
        self.textBrowser.setText(f"<h3>红色物料上下阈值:{lower_red} --> {higher_red}</h3>"
                                 f"<h3>绿色物料上下阈值:{lower_green} --> {higher_green}</h3>"
                                 f"<h3>蓝色物料上下阈值:{lower_blue} --> {higher_blue}</h3>")
        lower_red = np.array([self.Red_down_H.value(),
                              self.Red_down_S.value(),
                              self.Red_down_V.value()])
        higher_red = np.array([self.Red_up_H.value(),
                               self.Red_up_S.value(),
                               self.Red_up_V.value()])
        lower_red2 = np.array([156,
                              self.Red_down_S.value(),
                              self.Red_down_V.value()])
        higher_red2 = np.array([180,
                               self.Red_up_S.value(),
                               self.Red_up_V.value()])
        lower_green = np.array([self.Green_down_H.value(),
                                self.Green_down_S.value(),
                                self.Green_down_V.value()])
        higher_green = np.array([self.Green_up_H.value(),
                                 self.Green_up_S.value(),
                                 self.Green_up_V.value()])
        lower_blue = np.array([self.Blue_down_H.value(),
                               self.Blue_down_S.value(),
                               self.Blue_down_V.value()])
        higher_blue = np.array([self.Blue_up_H.value(),
                                self.Blue_up_S.value(),
                                self.Blue_up_V.value()])

    def go_CV_thread(self):
        self.evt_run.run()

    def evt_small(self):
        self.showMinimized()

    def evt_close(self):
        sys.exit(app.exec_())

    def go_R_page(self):
        self.stackedWidget.setCurrentIndex(0)

    def go_G_page(self):
        self.stackedWidget.setCurrentIndex(2)

    def go_B_page(self):
        self.stackedWidget.setCurrentIndex(1)

    def go_Red_down_Page(self):
        self.R_W.setCurrentIndex(0)

    def go_Red_up_Page(self):
        self.R_W.setCurrentIndex(1)

    def go_Green_down_Page(self):
        self.G_W.setCurrentIndex(0)

    def go_Green_up_Page(self):
        self.G_W.setCurrentIndex(1)

    def go_Blue_down_Page(self):
        self.B_W.setCurrentIndex(0)

    def go_Blue_up_Page(self):
        self.B_W.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = mainWin()
    main_win.show()
    sys.exit(app.exec_())
