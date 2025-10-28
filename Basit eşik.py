# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:02:01 2022

@author: i
"""

import cv2

resim = cv2.imread("scientistgri.jpg",0)

#ret eşik değerini ve maxı döndürücek
ret, resim_thresh = cv2.threshold(resim,180,255,cv2.THRESH_BINARY)

cv2.namedWindow("resim",cv2.WINDOW_NORMAL)
cv2.namedWindow("resim_thresh",cv2.WINDOW_NORMAL)

cv2.imshow("resim",resim)
cv2.imshow("resim_thresh",resim_thresh)
cv2.waitKey()
cv2.destroyAllWindows()