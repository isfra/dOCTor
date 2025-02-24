import os
import numpy as np
import GuiClean as GC
from openpyxl import Workbook, load_workbook
import shutil
import time

#env_path="/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/nuovi dati"
#model_file ="/home/eugenio/workspace/documets di pasqualino/Documents"#placcaultras"#4.7-1"#/4.4-12"#4.3/4.3-1.1"#modello2-4e-r0-all08"#modello2-mozzino-guida"#modello2e4r-5-pure"#modello2-4epoche-reg000001"#placcaultras"#modello2"#

gui=GC.Main()
'''
wb=load_workbook(os.path.dirname(os.path.realpath('__file__')) + '/NonMaceLBI.xlsx')#"/globalLBI.xlsx")

for sheet in wb:
    print(sheet.title)

data=wb['Sheet1']
keys={}


i=1
for cell in  data[1]:
    keys[cell.value]=i
    print(i)
    i+=1
    
print(keys)
'''
database = 'E:\Pazienti CLIMA' #'D:\\Franz\\oct_software\\data\\New OCT VS NIRS'    #'/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/pazienti mace'
#database='/media/fisici/Hitachi/Pazienti clima fra'
#database='/media/eugenio/Hitachi/Pazienti CLIMA'

#years= next(os.walk(database))[1]

errors_file= open(os.path.dirname(os.path.realpath('__file__')) + "/errors.txt","w+")
completed_file=open(os.path.dirname(os.path.realpath('__file__')) + "/completed.txt","w+")

patients = next(os.walk(database))[1]
#years= next(os.walk(database))[1]

#for year in years:

    #patients = next(os.walk(os.path.join(database, year)))[1]

for patient in patients :
    #print(os.path.join(database,year, patient))
    '''
    if  not next(os.walk(os.path.join(database,patient)))[1]:
        data_path=os.path.join(database,patient,'PB1')
        os.mkdir(data_path)
        #print(os.listdir(os.path.join(database, patient)))
        for file in os.listdir(os.path.join(database,patient)):
            #print(file)
            if not (file == 'PB1') :
                shutil.move(os.path.join(database,year,patient,file) , os.path.join(data_path,file))
    
    '''
    #for pb in next(os.walk(os.path.join(database,year,patient)))[1]:
    data_path=os.path.join(database,patient)
    ''' 
    n=0
    for file in os.listdir(data_path):
        if '.oct' in file:
            n+=1
    if n>1:
        print(data_path)
        
       for file in os.listdir(data_path):
            if '.oct' in file:
                data_path2=os.path.join(database,year,patient,'PB2')
                os.mkdir(data_path2)
                shutil.move(os.path.join(data_path,file) , os.path.join(data_path2,file))
                break
    '''


    print(data_path)
    gui.import_pullback_callback(data_path)
    try:
        gui.analyze_all()
    except:
        errors_file.write(data_path)
        errors_file.write("\n")
        print('ERROR IN ANALYSIS!!!!___________', data_path)
        pass

    '''    
    if not(os.path.exists(data_path+'/pullback.npy')) :
        
        gui.import_pullback_callback(data_path)
        
        gui.update_idletasks()
        gui.update()
        
        try:
            gui.analyze_all()
        except:
            errors_file.write(data_path)
            errors_file.write("\n")
            print('ERROR IN ANALYSIS!!!!___________' ,data_path)
            pass    
    '''


    '''
    ex=1
    try:
        guide = np.load(os.path.join(data_path, 'GuideArtifact.npy') )
    except:
        guide=np.zeros([2,2])
        pass
    
    #if not os.path.exists(data_path+'/Prediction.npy'):
    if (not sum(sum(guide))==0) and not os.path.exists(data_path+'/Prediction.npy'):       
        print('ERROR')
        errors_file.write(data_path)
        errors_file.write("\n")
    '''

    '''
    if not os.path.exists(data_path+'/GuideArtifact.npy'):#'/Prediction.npy'):

        print(data_path)

        #gui.mainloop()
        #gui.current_section_canvas.delete('angle_margin')

        #gui.current_section_canvas.show_calibration()

        #gui.update_idletasks()
        #gui.update()
        #time.sleep(2)
        #gui.mark_deep_plaque_n(depth=120)
        #gui.load_labels_classification()
        #gui.prediction_canvas.update(0)
        #gui.update_idletasks()
        #gui.update()
        #time.sleep(5)
        #gui.mark_deep_plaque_n(depth=100)

        #gui.load_labels_classification ()
        #gui.prediction_canvas.update(0)
        #gui.update_idletasks()
        #gui.update()
        #time.sleep(5)

        try:

            gui.import_pullback_callback(data_path)
            gui.analyze_all()
            '''
            for inter in gui.interv_of_pertinence:
                for f in range(inter[0], inter[1]):
                    gui.current_section_canvas.calculate_fct_cart(f)
            GC.save_list(gui.fct, gui.pullback_dir + '/Fct.txt')
            gui.all_fct = gui.all_fct.astype(np.int)
            np.save(gui.all_fct_path, gui.all_fct)
            '''

            with open(os.path.dirname(os.path.realpath('__file__')) + "/completed_fc.txt","a") as completed_file :
                completed_file.write(data_path)
                completed_file.write("\n")

            #gui.analyze_LBI(superf=0)
            #gui.analyze_LBI(superf=1)
        except:
            with open(os.path.dirname(os.path.realpath('__file__')) + "/errors.txt","a") as errors_file:
                errors_file.write(data_path)
                errors_file.write("\n")
            print('ERROR IN ANALYSIS!!!!___________' ,data_path)
            pass
    '''
errors_file.close()
completed_file.close()