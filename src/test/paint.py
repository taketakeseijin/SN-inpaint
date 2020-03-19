# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
# sx, syは線の始まりの位置
sx, sy = 0, 0
# マウスの操作があるとき呼ばれる関数
IMAGE_SIZE = 128


def callback(event, x, y, flags, param):
    global img, sx, sy, mask
    # マウスの左ボタンがクリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy = x, y
    # マウスの左ボタンがクリックされていて、マウスが動いたとき
    if flags == cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, (sx, sy), (x, y), (255, 255, 255), 5)
        cv2.line(mask, (sx, sy), (x, y), (255, 255, 255), 5)
        sx, sy = x, y


def main(target_name, save_name):
    # 画像を読み込む
    global img, mask
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    cv2.imwrite("./paintworks/black.jpg", mask)
    mask = cv2.imread("./paintworks/black.jpg")

    img = cv2.imread(target_name)
    # ウィンドウの名前を設定
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # コールバック関数の設定
    cv2.setMouseCallback("img", callback)
    while True:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        # Escキーを押すと終了
        if k == 27:
            sys.exit()
        # sを押すと画像を保存
        if k == ord("s"):
            cv2.imwrite(save_name, mask)
            break
