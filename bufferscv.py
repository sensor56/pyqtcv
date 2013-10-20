#!/usr/bin/python
# -*- coding: utf-8 -*-

# pyqtcv
# modules python implémentant plusieurs fonctions utiles pour le traitement d'image avec OpenCV 
# et PyQt

# Par X. HINAULT - Tous droits réservés - GPLv3
# 2012 - 2013 - www.mon-club-elec.fr

# Buffers OpenCV 
RGB = None
R, G, B = None, None, None
Gray = None

Memory=None

Trans16S3C, Trans16S3C1, Trans16S3C2 = None, None, None
Trans16S1C, Trans16S1C1, Trans16S1C2 = None, None, None
Trans8U3C, Trans8U3C1, Trans8U3C2=None, None, None
Trans8U1C, Trans8U1C1, Trans8U1C2= None, None, None
Trans32F3C=None

# les buffers sont communs à tous les modules qui les utilisent 

# autres objets communs utiles
qImage=None
qPixmap=None

webcam=None 

# gsvideo
pipeline=None
sink=None 

