#!/usr/bin/python
# -*- coding: utf-8 -*-

# pyqtcv
# modules python implémentant plusieurs fonctions utiles pour le traitement d'image avec OpenCV 
# et PyQt

# Par X. HINAULT - Tous droits réservés - GPLv3
# 2012 - 2013 - www.mon-club-elec.fr

# --- importation des modules utiles ---
from PyQt4.QtGui import *
from PyQt4.QtCore import * # inclut QTimer..

import os,sys
from cv2 import * # importe module OpenCV - cv est un sous module de cv2

import buffers # importe les buffers OpenCV utiles pour pyqtcv
# voir : http://docs.python.org/2/faq/programming.html#how-do-i-share-global-variables-across-modules 

# --- classes utiles (à placer avant la classe principale) --- 
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

#----- fonction d'initialisation / création des buffers OpenCV utiles	
def allocate(widthIn, heightIn):
	
	"""
	global Buffer
	global BufferR, BufferG, BufferB
	global BufferGray
	
	global Trans16S3C, Trans16S3C1, Trans16S3C2
	global Trans16S1C, Trans16S1C1, Trans16S1C2
	global Trans8U3C, Trans8U3C1, Trans8U3C2
	global Trans8U1C, Trans8U1C1, Trans8U1C2
	global Trans32F3C
	"""
	
	mySize=(widthIn, heightIn)
	
	#--- création d'un buffer principal RGB utilisé par les fonctions 
	buffers.RGB=cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 3) # buffer principal 3 canaux 8 bits non signés - RGB --
	
	#--- crée 3 buffers 1 canal 8 bits non signés = 1 canal par couleur 
	buffers.R = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) #1 canal - canal rouge
	buffers.G = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) # 1 canal - canal vert 
	buffers.B = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) # 1 canal - canal bleu 

	#BufferA = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // 1 canal - canal transparence 

	#--- crée un buffer 1 canal 8 bits non signés = niveaux de gris 
	buffers.Gray = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) # 1 canal - niveaux de gris 		
	
	#----- Buffers de travail --- 
	
	#-- 16S 3C ---
	# crée une image Trans de travail, 3 canaux 16bits Signés
	buffers.Trans16S3C = cv.CreateImage(mySize, cv.IPL_DEPTH_16S, 3) # crée image Ipl 16bits Signés - 3 canaux
	buffers.Trans16S3C1 = cv.CreateImage(mySize, cv.IPL_DEPTH_16S, 3) # crée image Ipl 16bits Signés - 3 canaux
	buffers.Trans16S3C2 = cv.CreateImage(mySize, cv.IPL_DEPTH_16S, 3) # crée image Ipl 16bits Signés - 3 canaux
	
	#-- 16S 1C ---		
	# crée une image Trans de travail, 1 canal 16 bits Signés
	buffers.Trans16S1C = cv.CreateImage(mySize, cv.IPL_DEPTH_16S, 1) # crée image Ipl 16bits Signés - 1 canal
	buffers.Trans16S1C1 = cv.CreateImage(mySize, cv.IPL_DEPTH_16S, 1) # crée image Ipl 16bits Signés - 1 canal
	buffers.Trans16S1C2 = cv.CreateImage(mySize, cv.IPL_DEPTH_16S, 1) # crée image Ipl 16bits Signés - 1 canal
	
	#-- 8U - 3C ---
	# crée une image Trans de travail, 3 canaux 8bits non signés (Unsigned)
	buffers.Trans8U3C = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 3) # crée image Ipl 8bits non Signés - 3 canaux
	buffers.Trans8U3C1 = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 3) # crée image Ipl 8bits non Signés - 3 canaux
	buffers.Trans8U3C2 = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 3) # crée image Ipl 8bits non Signés - 3 canaux
	
	# -- 8U - 1C ---
	# crée une image Trans de travail, 1 canal 8bits non signés (Unsigned)
	buffers.Trans8U1C = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) # crée image Ipl 8bits non Signés - 1 canal
	buffers.Trans8U1C1 = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) # crée image Ipl 8bits non Signés - 1 canal
	buffers.Trans8U1C2 = cv.CreateImage(mySize, cv.IPL_DEPTH_8U, 1) # crée image Ipl 8bits non Signés - 1 canal
	
	#-- 32F - 1C ---
	# crée une image Trans de travail, 3 canaux 32bits à virgule flottante (Float)
	buffers.Trans32F3C = cv.CreateImage(mySize, cv.IPL_DEPTH_32F, 3) # crée image Ipl 32F (à virgule - signés) - 3 canaux
	
	
##-- fin de la déclaration des buffers utiles 

#---- conversion gray d'une image RGB 3 canaux ----
def gray(iplImgIn):

	cv.CvtColor(iplImgIn, buffers.Gray, cv.CV_RGB2GRAY) # bascule l'image 3 canaux en niveaux de gris dans le buffer Gray 
	
	cv.CvtColor(buffers.Gray, iplImgIn, cv.CV_GRAY2RGB) # rebascule le buffer Gray en RGB 
	# les 3 canaux du buffer RGB sont identiques = l'image 3 canaux est en niveaux de gris 
	
	# la copie est conservée dans le buffer Gray
	
	#return(iplImgIn) # pas indispensable - l'image reçue est modifiée 

##--- fin gray()

#---- effet miroir vertical / horizontal ou les 2 ---
VERTICAL=0
HORIZONTAL=1
BOTH=2

def flip (iplImgIn, flipModeIn): 
	
	cv.Flip(iplImgIn, iplImgIn, flipModeIn) # flip vertical : applique effet miroir vertical au Buffer 
	
##--- fin flip()

#-- fonction d'inversion d'une image RGB -- 
def invert(iplImgIn):
	myRGB = cv.CV_RGB(255, 255, 255) # crée scalaire de 3 valeurs
	cv.SubRS(iplImgIn, myRGB,iplImgIn,None) # soustraction inverse du scalaire pour tous les pixels

##--- fin invert()
