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
import time # pour millis

import bufferscv as buffers # importe les buffers OpenCV utiles pour pyqtcv
# voir : http://docs.python.org/2/faq/programming.html#how-do-i-share-global-variables-across-modules 

#-- pour capture gsvideo 
import pygst # gsvideo
pygst.require('0.10')
import gst

import gobject # indispensable pour bon fonctionnement signaux pygst - GObject est utilisé comme base des plugins 
gobject.threads_init() # idem.. 

# math 
import numpy as np

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

# -- fin class IplToQImage

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

""" # pour éviter problemes : rester avec qImage et qPixmap self. 
#--- affichage d'un iplImage dans un QLabel 
def display(iplImageIn, qLabelIn):
	buffers.qImage=IplToQImage(iplImageIn) # récupère l'IplImage dans un QImage - obligé utiliser qImage différent
	buffers.qPixmap=QPixmap.fromImage(buffers.qImage) # récupère le QImage dans un QPixmap
	qLabelIn.setPixmap(buffers.qPixmap) # affiche l'image dans le label 
## fin display 
"""

#---- conversion gray d'une image RGB 3 canaux ----
def gray(*arg):
	
	if len(arg)==0: # buffer RGB par defaut
		iplImgIn=buffers.RGB
	elif len(arg)==1: # si iplImage passe en parametre
		iplImgIn=arg[0]
	
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

#-- fonction de floutage d'une image RGB -- 
def smooth(*arg):
	
	if len(arg)==0: # buffer RGB par defaut
		cv.Smooth(buffers.RGB, buffers.RGB,cv.CV_GAUSSIAN, 5,0,0,0) # applique flou gaussien avec noyau de taille voulue
	elif len(arg)==1:
		iplImgIn=arg[0]
		# cv.Smooth(src, dst, smoothtype=CV_GAUSSIAN, size1=3, size2=0, sigma1=0, sigma2=0) → None
		cv.Smooth(iplImgIn, iplImgIn,cv.CV_GAUSSIAN, 5,0,0,0) # applique flou gaussien avec noyau de taille voulue

##--- fin smooth()

BINARY=cv.CV_THRESH_BINARY
BINARY_INV=cv.CV_THRESH_BINARY_INV
TRUNC=cv.CV_THRESH_TRUNC
TOZERO=cv.CV_THRESH_TOZERO
TOZERO_INV=cv.CV_THRESH_TOZERO_INV

#-- fonction de seuillage d'une image RGB -- 
def threshold(*arg):
	#(seuil) : buffer.RGB et BINARY par defaut
	#(image, seuil) : BINARY par defaut
	#(image, seuil, mode)
	
	if len(arg)==1: #(seuil) : buffer.RGB et BINARY par defaut
		valeur=iplImgIn=arg[0]
		#cv.Threshold(src, dst, threshold, maxValue, thresholdType) → None
		cv.Threshold(buffers.RGB, buffers.RGB, valeur, 255, cv.CV_THRESH_BINARY) # applique seuillage binaire
	elif len(arg)==2: #(image, seuil) : BINARY par defaut
		iplImgIn=arg[0]
		valeur=arg[1]
		# cv.Smooth(src, dst, smoothtype=CV_GAUSSIAN, size1=3, size2=0, sigma1=0, sigma2=0) → None
		cv.Threshold(iplImgIn, iplImgIn, valeur, 255, cv.CV_THRESH_BINARY) # applique seuillage binaire
	elif len(arg)==3: #(image, seuil, mode)
		iplImgIn=arg[0]
		valeur=arg[1]
		modeIn=arg[2]
		cv.Threshold(iplImgIn, iplImgIn, valeur, 255, modeIn) # applique seuillage binaire

##--- fin threshold()

#def mixerRGB (iplImgRGBIn, coeffRIn, coeffGIn, coeffBIn, canalOut, grayOut, debugIn):
def mixerRGB (iplImgRGBIn, coeffRIn, coeffGIn, coeffBIn, grayOut):
	# iplImgRGBIn : image RGB In (iplImage)
	# coeffRIn, coeffGIn, coeffBIn : coeff à appliquer aux canaux RGB (float)
	# canalOut : canal de sortie à utiliser (int)
	# grayOut : drapeau de sortie mixer Gray
	# debugIn : drapeau debug
	
	# la fonction mixerRGB implémente un algorithme du logiciel The Gimp appelé "mixeur de canaux"
	# un des canaux RGB est choisi comme canal de sortie (R par défaut)
	# chaque pixel est recalculé en intégrant une certaine proportion des autres canaux selon : 
	# R = (R x coeff R) + (G x coeff G) + (B x coeff B)

	float(coeffRIn)
	float(coeffGIn)
	float(coeffBIn)
	
	# --- conversion de l'image source en 16S 
	# opencv_core.cvConvertScale(iplImgSrc, iplImgSrc16S, 256.0, -32768); // convertit 8U en 16S 
	cv.ConvertScale(iplImgRGBIn, buffers.Trans16S3C, 1, 0); # convertit 8U en 16S mais en gardant les valeurs 8U
	
	#---- application d'un coefficient à chaque canal ---- 
	# calcul séparément pour chaque pixel R x coeff R , G x coeff G, B x coeff B
	  
	# --- création de 3 matrices de float contenant les coeff RGB à appliquer :
	cv.Set(buffers.Trans32F3C, cv.CV_RGB(coeffRIn, coeffGIn, coeffBIn)); # remplit tous les pixels du IplImage avec le scalaire (coeffB, coeffG, coeffR)
	#pour pouvoir stocker des float, il faut que l'image soit une 32F !!
	# attention coeff R et B PAS inversés  ici 

	cv.Mul(buffers.Trans16S3C, buffers.Trans32F3C, buffers.Trans16S3C, 1); # multiplie les 2 images 
	#ce qui réalise pour chaque pixel : R x coeff R , G x coeff G, B x coeff B
	#-- nb ; pour la multiplication avec des float, il faut que l'image soit une 32F
	
	# -- dissociation des canaux de l'image source en 3 IplImages mono-canal
	#cv.Split(iplImgSrc, iplImgR, null, null, null); 
	#-- on utilise Trans16S1C pour R , Trans16S1C1 pour G et Trans16S1C pour B
	cv.Split(buffers.Trans16S3C, buffers.Trans16S1C2, buffers.Trans16S1C1, buffers.Trans16S1C, None); # extrait les 3 canaux R G B : attention inversion RGB  

	#-- calcul du canal de sortie rouge par sommation des 3 canaux avec coefficients 
	#-- correspond à l'addition R = (R x coeff R) + (G x coeff G) + (B x coeff B)
	cv.Add(buffers.Trans16S1C, buffers.Trans16S1C1, buffers.Trans16S1C, None); # additionne les 2 IplIMage dans un troisième - ici, réalise R = (R x coeff R) + (G x coeff G)
	cv.Add(buffers.Trans16S1C, buffers.Trans16S1C2, buffers.Trans16S1C, None); # additionne les 2 IplIMage dans un troisième - ici, réalise R = (R x coeff R) + (G x coeff G)

	#conversion d'un Objet Ipl16S en 8U mais sans changer la valeur
	# cv.ConvertScale(iplImg16S,iplImg8U,(1.0/256.0),128);  conversion 16S vers 8U avec adaptation valeur pleine échelle 
	cv.ConvertScale(buffers.Trans16S1C,buffers.Trans8U1C,1,0); # sans changer la valeur

	#-- récupère les canaux G et B de départ --- 
	cv.Split(iplImgRGBIn, buffers.Trans8U1C2, buffers.Trans8U1C1, None, None); # extrait les 2 canaux G B : attention inversion RGB  
	# on ne récupère pas ici le canal R car on l'a recalculé

	#-- reconstruction des canaux de l'image source à partir des 3 IplImages mono-canal  
	if grayOut :
		  # sort les 3 canaux Idem 
		  cv.Merge(buffers.Trans8U1C,buffers.Trans8U1C,buffers.Trans8U1C, None, iplImgRGBIn); # reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB 
		  #Les canaux B et G sont inchangés et le canal R = = (R x coeff R) + (G x coeff G) + (B x coeff B)

	else : # si pas sortie en niveau de gris 
		cv.Merge(buffers.Trans8U1C2,buffers.Trans8U1C1,buffers.Trans8U1C, None, iplImgRGBIn); # reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB 
		# Les canaux B et G sont inchangés et le canal R = = (R x coeff R) + (G x coeff G) + (B x coeff B)
		
		
	#return iplImgIn

# fin mixerRGB


#-- redimensionnement d'une image vers une autre - l'image destination sert de reference
def resize(iplImgSrcIn, iplImgDestIn):
	cv.Resize(iplImgSrcIn,iplImgDestIn, cv.CV_INTER_LINEAR) # redimensionne image
##-- fin resize

#-- copie image
def copyTo(iplImgSrcIn, iplImgDestIn):
	cv.Copy(iplImgSrcIn, iplImgDestIn) # copie l'image de départ image destination
##-- fin copyTo

""" # probleme avec self
# fonction de sélection fichier
def selectFile(currentPathIn):
	#self.filename=QFileDialog.getOpenFileName(self, 'Ouvrir fichier', os.getenv('HOME')) # ouvre l'interface fichier
	#self.filename=QFileDialog.getOpenFileName(self, 'Ouvrir fichier', QDir.currentPath()) # ouvre l'interface fichier
	# getOpenFileName ouvre le fichier sans l'effacer et getSaveFileName l'efface si il existe 
	
	#-- ouverture fichier en memorisant dervnier fichier ouvert 
	if currentPathIn=="":
		#self.filename=QFileDialog.getOpenFileName(self, 'Ouvrir fichier', os.getenv('HOME')) # ouvre l'interface fichier - home par défaut
		filenameOut=QFileDialog.getOpenFileName(self, 'Ouvrir fichier', QDir.currentPath()) # ouvre l'interface fichier - chemin courant par défaut
	else:
		info=QFileInfo(currentPathIn) # définit un objet pour manipuler info sur fichier à partir chaine champ
		print info.absoluteFilePath() # debug	
		filenameOut=QFileDialog.getOpenFileName(self, 'Ouvrir fichier', info.absoluteFilePath()) # ouvre l'interface fichier - à partir chemin 
	
	return filenameOut
"""

########## Webcam #########

#----------- fonctions OpencCV ------------
def initWebcam(indexCamIn,widthCaptureIn, heightCaptureIn, fpsIn):
	buffers.webcam=cv.CaptureFromCAM(indexCamIn) # déclare l'objet capture sans désigner la caméra - remplace CreateCameraCapture		print (self.capture) # debug
	cv.SetCaptureProperty(buffers.webcam,cv.CV_CAP_PROP_FRAME_WIDTH,widthCaptureIn) # fixe largeur de capture
	cv.SetCaptureProperty(buffers.webcam,cv.CV_CAP_PROP_FRAME_HEIGHT,heightCaptureIn) # fixe hauteur de capture
	#cv.SetCaptureProperty(self.webcam,cv.CV_CAP_PROP_FPS,30) # fixe le framerate hardware- pas toujours supporté par webcam
	return buffers.webcam

#----- fonction de classe : capture d'une image Ipl à partir de la webcam
def captureImage(*args): # capture une nouvelle image issue de la webcam
	if len(args)==0: 
		iplImgCapture = cv.QueryFrame(buffers.webcam) # frame est une iplImage - format OpenCV
	elif len(args)==1: 
		webcamIn=args[0]
		iplImgCapture = cv.QueryFrame(webcamIn) # frame est une iplImage - format OpenCV
	
	return iplImgCapture

#----------- fonctions GSVideo -------------
def initWebcamGS(indexCamIn,widthCaptureIn, heightCaptureIn, fpsIn):
	
	print ('''v4l2src device=/dev/video'''+str(indexCamIn) 
	+'''! video/x-raw-rgb,width='''+str(widthCaptureIn)+''',height='''+str(heightCaptureIn)
	+''',framerate='''+str(fpsIn)+'''/1 ! appsink name=sink emit-signals=true'''
	)
	
	# initialisation pipeline style "ligne de commande"
	buffers.pipeline = gst.parse_launch(
	'''v4l2src device=/dev/video'''+str(indexCamIn) 
	+''' ! video/x-raw-rgb,width='''+str(widthCaptureIn)
	+''',height='''+str(heightCaptureIn)+''',framerate='''+str(fpsIn)+'''/1 ! appsink name=sink emit-signals=true'''
	#'''v4l2src device=/dev/video0 ! video/x-raw-rgb,width=320,height=240,framerate=30/1 ! appsink name=sink emit-signals=true''' # exemple
	# laisser espace avant/apres les ! 
	) # fin commande initalisation pipeline
	
	buffers.sink = buffers.pipeline.get_by_name('sink')
	
	# lancement du pipeline
	buffers.pipeline.set_state(gst.STATE_PLAYING)
	
	# connexion signal avec fonction lecture data
	#buffers.sink.connect('new-buffer', readBuffer) 

# fonction lecture des data images
def readDataBufferGS(bufIn): # appelée par le code principal - la fonction reçoit le buffer
	
	# numpy des données
	pixels = np.frombuffer(bufIn.data, dtype=np.uint8)
	pixels=pixels.reshape((320,240,3))

	#print type(self.pixels)
	#print self.pixels
	#print self.pixels.shape
	
	# numpy vers IplImage => donne acces aux fonctions Opencv: 
	myIplImage = cv.CreateImageHeader((pixels.shape[0], pixels.shape[1]), cv.IPL_DEPTH_8U, 3)
	cv.SetData(myIplImage, pixels.tostring(),pixels.dtype.itemsize * 3 * pixels.shape[0])
	# adaptation de : http://stackoverflow.com/questions/11528009/opencv-converting-from-numpy-to-iplimage-in-python
	return myIplImage

def captureImageGS(sinkIn):
	
	# affichage de test 
	buf = sinkIn.emit('pull-buffer') # attend l'émission du signal... 
	# si la fonction est appelée par signal = immédiat - sinon correspond au délai timer...
	
	return readDataBufferGS(buf) # renvoie iplImage à partir data 

########## temps ##########
def millis():
	return int(round(time.time() * 1000))

def micros():
	return(int(round(time.time() * 1000000))) # microsecondes de l'horloge système

########## blobs #########

#=========== Classe Blob =================
class Blob: # cette classe rassemble dans un meme objet toutes les caractéristiques utiles d'un blob = pourtour de forme
	
	def __init__(self, indiceContourIn, contourIn, areaIn, lengthArcIn, lengthIn, centroidPointIn, rectIn, pointsIn): 
		# indiceContourIn, float areaIn, float lengthArcIn, float lengthIn, Point centroidIn, Rectangle rectIn, Point[] pointsIn
		self.indiceContour = indiceContourIn # indice du blob
		self.contour=contourIn # le cvSeq du contour
		self.area = areaIn # aire du blob
		self.lengthArc=lengthArcIn # perimetre du blob
		self.length=lengthIn # nombre de points du blob
		self.centroidPoint=centroidPointIn # centre du blob
		self.rect=rectIn # rectangle encadrant
		self.points=pointsIn # tableau de points du blob
		

#========= Fonctions utilisant les Blobs ==========
"""
    detectBlobs() w
    drawBlobs() w
    drawCentroidBlobs() w
    drawRectBlobs() w 
    selectBlobs() w
    selectBallBlobs() w
    keypointsSBD() w 
"""
#-- fonction blobs() : renvoie la liste des formes detectees -- 
def detectBlobs(*arg): 
	
	# gestion des arguments
	if len(arg)==0: # buffer RGB par defaut
		iplImgIn=buffers.RGB
	elif len(arg)==1: # si iplImage passe en parametre
		iplImgIn=arg[0]
	
	# chargement dans buffer gray
	gray(iplImgIn) # conversion en niveau de gris
	
	# extraction de contours
	storage = cv.CreateMemStorage(0)
	contours = cv.FindContours(buffers.Gray, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
	
	# creation d'une list d'objets Blob
	blobsList=list()
	
	while contours: # defile les contours obtenus 
		
		# indice du blob
		indiceContour=len(blobsList) # indice du blob correspond a la taille courante de la liste de blob
		
		# CvSeq du blob = le contour courant 
		contour=contours # utilise le contour courant 
		
		# area=0 # aire du blob
		area=cv.ContourArea(contours) # l'aire du contour courant - nb : cf il es possible de tenir compte de l'orientation... 
		
		#lengthArc=0 # perimetre du blob
		lengthArc=cv.ArcLength(contours) # le perimetre du contour courant 
		
		# nombre de points du blob courant
		length=len(contours) # le nombre de points du contour courant
		
		#centroidPoint=[(0,0)] # centre du blob
		moments=cv.Moments(contours) # calcule les moments du contour courant 
		m00 = cv.GetSpatialMoment(moments, 0, 0) # recupere moment sous forme sous forme m00, m10.. 
		m10 = cv.GetSpatialMoment(moments, 1, 0)
		m01 = cv.GetSpatialMoment(moments, 0, 1)
		centroid_x=int(m10/m00) # calcule x du centre
		centroid_y=int(m01/m00) # calcule y du centre
		centroidPoint=[(centroid_x, centroid_y)] # centre du blob
		
		#rect=[(0,0),(0,0)] # rectangle entourant le blob - coin sup gauche - coin inf droit 
		bound_rect = cv.BoundingRect(list(contours)) # récupère le rectangles encadrant le contour courant
		pt1 = (bound_rect[0], bound_rect[1]) # récupère le 1er point du rectangle encadrant
		pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3]) # récupère le 2ème point du rectangle encadrant 
		rect=[] # list des points
		rect.append(pt1) # ajoute le point au tableau
		rect.append(pt2) # ajoute le 2ème point au tableau
		
		# list de spoints du contour
		points=list(contours) # liste des points du contour courant
		
		blobsList.append(Blob(indiceContour, contour, area, lengthArc, length, centroidPoint, rect, points)) # ajoute un objet Blob à la list
		
		
		contours = contours.h_next() # passe au contour suivant - remarquer la forme contours=contours.h_next et pas seulement contours.h_next
	
	return blobsList # renvoie la liste d'objet blobs

#-- fonction drawBlobs : dessine les blobs sur une image
def drawBlobs(blobsIn, iplImgIn):
	#while blobsIn:
	for blob in blobs : # defile les Objets blobs pyqtcv obtenus
		# cv.DrawContours(img, contour, external_color, hole_color, max_level, thickness=1, lineType=8, offset=(0, 0))
		cv.DrawContours(iplImgIn, blob.contour,cv.CV_RGB(255,255,0),cv.CV_RGB(255,255,0),1) # dessine tous les contours - laissés vide 
		#cv.DrawContours(iplImgIn, blobsIn,cv.CV_RGB(255,255,0),cv.CV_RGB(0,255,0),1,-1) # dessine tous les contours - remplis - utilise CPU...
