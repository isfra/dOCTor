import os
import numpy as np
import GuiClean as GC
from openpyxl import Workbook, load_workbook
import shutil

import matplotlib.pyplot as plt
from PIL import Image

#database='/media/fisici/Hitachi/Pazienti CLIMA'
#database='/media/fisici/Hitachi/Pazienti clima fra'
database='D:\Franz\oct_software\data\OCTvsNIRS'

lp = 104
thr = 500

NonMace= ["Santarcangelo","Alessandri","Buzzanca","Cristini","Milana","Cusmai","Milito","Giacomini","Daltin","D'ettore",
          "Monti","Santone","Canarile","D'Auria","Siniscalchi","Gelfusa","Proietti","Russo","Bruttini","Caselli","Caldarini","Pette",
          "Gambini","Campanella","Ippoliti","Saldicco","Costanzo","Savastano","Ahmed","Abruzzo","Zeppi","Tantalo","Russo",
          "Ercoli","Missagia","Mia"]

chem_path=database +'/Chemiograms'



fig,ax = plt.subplots(11,5)
i=0
for file in os.listdir(chem_path):
    #if np.any(np.array([patient in file for patient in NonMace])) and ('Low' in file): 
    if  ('PB' in file) and ('High' in file): 
         
        with open(os.path.join(chem_path,file),'rb') as f:
            image=Image.open(f)
            ax[i//5][i%5].imshow(image)
            i+=1

fig.savefig(chem_path+ '/37Mh.png')
fig.show()