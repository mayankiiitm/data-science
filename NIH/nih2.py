import pandas as pd
import numpy as np
import cv2
from glob import glob
import os
labels = pd.read_csv('sample_labels.csv')

labels = labels[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]
#create new columns for each decease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']
for pathology in pathology_list :
    labels[pathology] = labels['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
#remove Y after age
labels['Age']=labels['Patient Age'].apply(lambda x: x[:-1]).astype(int)

labels['Nothing']=labels['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)

images=glob('images/*')

def proc_images():
    NoFinding = "No Finding" #0
    Consolidation="Consolidation" #1
    Infiltration="Infiltration" #2
    Pneumothorax="Pneumothorax" #3
    Edema="Edema" # 7
    Emphysema="Emphysema" #7
    Fibrosis="Fibrosis" #7
    Effusion="Effusion" #4
    Pneumonia="Pneumonia" #7
    Pleural_Thickening="Pleural_Thickening" #7
    Cardiomegaly="Cardiomegaly" #7
    NoduleMass="Nodule" #5
    Hernia="Hernia" #7
    Atelectasis="Atelectasis"  #6 
    RareClass = ["Edema", "Emphysema", "Fibrosis", "Pneumonia", "Pleural_Thickening", "Cardiomegaly","Hernia"]
    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 64
    HEIGHT = 64
    for img in images:
        base = os.path.basename(img)
        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        # Read and resize image
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        # Labels
        if NoFinding in finding:
            finding = 0
            y.append(finding)        
        elif Consolidation in finding:
            finding = 1
            y.append(finding)    
        elif Infiltration in finding:
            finding = 2
            y.append(finding)           
        elif Pneumothorax in finding:
            finding = 3
            y.append(finding)
        elif Edema in finding:
            finding = 7
            y.append(finding)
        elif Emphysema in finding:
            finding = 7
            y.append(finding)
        elif Fibrosis in finding:
            finding = 7
            y.append(finding) 
        elif Effusion in finding:
            finding = 4
            y.append(finding)             
        elif Pneumonia in finding:
            finding = 7
            y.append(finding)   
        elif Pleural_Thickening in finding:
            finding = 7
            y.append(finding) 
        elif Cardiomegaly in finding:
            finding = 7
            y.append(finding) 
        elif NoduleMass in finding:
            finding = 5
            y.append(finding) 
        elif Hernia in finding:
            finding = 7
            y.append(finding) 
        elif Atelectasis in finding:
            finding = 6
            y.append(finding) 
        else:
            finding = 7
            y.append(finding)
    return x,y

X,y = proc_images()

ix=images[0:10]

iy=[]
for img in ix:
    base = os.path.basename(img)
    finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
    iy.append(finding)