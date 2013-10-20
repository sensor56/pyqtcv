pyqtcv
======

pyqtcv est une librairie personnelle rassemblant des classes et fonctions utiles pour l'utilisation d'OpenCV avec PyQt, facilitant la mise en place d'interfaces de traitement d'images fixes ou à partir d'un flux vidéo de webcam. 

Cette librairie est une ré-implémentation en PyQt de ma librairie JavacvPro écrite en Java pour Processing : 
http://www.mon-club-elec.fr/pmwiki_reference_lib_javacvPro/pmwiki.php

Cette migration permet de profiter de toute la puissance d'OpenCV et de JavacvPro, mais au sein d'une interface PyQt qui s'intègre parfaitement à l'environnement graphique du système. 
En terme de performance, un traitement de flux webcam 320x240 en temps réel obtenu à 100fps !

### Installation 

Soit dans un répertoire de son choix :   
cd /dir/to/use/   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/pyqtcv.py   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/bufferscv.py


L'appel se fait alors dans le code sous la forme :   
sys.path.insert(0,'/home/user') # si pas path système   
from pyqtcv import * # importe librairie perso comportant fonctions utiles pour utiliser opencv avec pyqt   
import bufferscv as buffers 

Soit dans répertoire Python système :   
cd /usr/lib/python2.7/dist-packages   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/pyqtcv.py   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/bufferscv.py


L'appel se fait alors dans le code sous la forme :   
from pyqtcv import * # importe librairie perso comportant fonctions utiles pour utiliser opencv avec pyqt   
import bufferscv as buffers 

### Classes disponibles 
IplToQImage(IplImage)--> QImage

### Fonctions disponibles 

