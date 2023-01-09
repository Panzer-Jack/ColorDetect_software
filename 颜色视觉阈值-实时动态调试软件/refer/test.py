import cv2
import numpy as np

frame = cv2.imread("all.png")
frame_width = frame.shape[0]
frame_height = frame.shape[1]
print(frame.shape)

lower_red = np.array([0, 43, 46])
higher_red = np.array([10, 255, 255])
lower_red2 = np.array([156, 43, 46])
higher_red2 = np.array([180, 255, 255])
lower_green = np.array([35, 43, 46])
higher_green = np.array([77, 255, 255])
lower_blue = np.array([100, 43, 46])
higher_blue = np.array([124, 255, 255])

while True:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv_frame, lower_red, higher_red)
    mask_green = cv2.inRange(hsv_frame, lower_green, higher_green)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, higher_blue)
    mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波
    mask_red = cv2.medianBlur(mask_red, 7)
    mask_blue = cv2.medianBlur(mask_blue, 7)
    # cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
    # cnts3, hierarchy3 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.circle(frame, (int(frame_width/2), int(frame_height/2)), 5, (255, 0, 0), 3)

    cv2.imshow("Test", frame)
    cv2.imshow("red", mask_red)
    cv2.imshow("green", mask_green)
    cv2.imshow("blue", mask_blue)
    if cv2.waitKey(1) == ord("q"):
        break

