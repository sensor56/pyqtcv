#!/usr/bin/python
# -*- coding: utf-8 -*-

# pyqtcv
# module python implémentant plusieurs fonctions utiles pour le traitement d'image avec OpenCV 
# et PyQt 

# Par X. HINAULT - Tous droits réservés - GPLv3
# 2012 - 2013 - www.mon-club-elec.fr

# --- importation des modules utiles ---
from PyQt4.QtGui import *
from PyQt4.QtCore import * # inclut QTimer..

import os,sys
from cv2 import * # importe module OpenCV - cv est un sous module de cv2


# --- classes utiles  --- 
class IplToQImage(QImage):
# IplQImage est une classe qui transforme un iplImage (OpenCV) en un QImage (Qt)
    """
http://matthewshotton.wordpress.com/2011/03/31/python-opencv-iplimage-to-pyqt-qimage/
A class for converting iplimages to qimages
"""
    def __init__(self,iplimage):
        # Rough-n-ready but it works dammit
        alpha = cv.CreateMat(iplimage.height,iplimage.width, cv.CV_8UC1)
        cv.Rectangle(alpha, (0, 0), (iplimage.width,iplimage.height), cv.ScalarAll(255) ,-1)
        rgba = cv.CreateMat(iplimage.height, iplimage.width, cv.CV_8UC4)
        cv.Set(rgba, (1, 2, 3, 4))
        cv.MixChannels([iplimage, alpha],[rgba], [
        (0, 0), # rgba[0] -> bgr[2]
        (1, 1), # rgba[1] -> bgr[1]
        (2, 2), # rgba[2] -> bgr[0]
        (3, 3) # rgba[3] -> alpha[0]
        ])
        self.__imagedata = rgba.tostring()
        super(IplToQImage,self).__init__(self.__imagedata, iplimage.width, iplimage.height, QImage.Format_RGB32)

# -- fin class IplQImage
