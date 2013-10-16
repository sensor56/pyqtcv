pyqtcv
======

pyqtcv est une librairie personnelle rassemblant des classes et fonctions utiles pour l'utilisation d'OpenCV avec PyQt, facilitant la mise en place d'interfaces de traitement d'images fixes ou à partir d'un flux vidéo de webcam. 

### Installation 

Soit dans un répertoire de son choix :   
cd /dir/to/use/   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/pyqtcv.py   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/buffers.py


L'appel se fait alors dans le code sous la forme :   
sys.path.insert(0,'/home/user') # si pas path système   
from pyqtcv import * # importe librairie perso comportant fonctions utiles pour utiliser opencv avec pyqt

Soit dans répertoire Python système :   
cd /usr/lib/python2.7/dist-packages   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/pyqtcv.py   
sudo wget -4 -N https://raw.github.com/sensor56/pyqtcv/master/buffers.py


L'appel se fait alors dans le code sous la forme :   
from pyqtcv import * # importe librairie perso comportant fonctions utiles pour utiliser opencv avec pyqt   
import buffers 

### Classes disponibles 
IplToQImage(IplImage)--> QImage

### Fonctions disponibles 

