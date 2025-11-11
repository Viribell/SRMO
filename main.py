from myModel import *
from myUtils import *

import numpy as np
import cv2
from PIL import Image, ImageTk

g_Model = None

g_ModelsFolderName = "models"
g_ModelName = "myModel_save_full"
g_ModelExt = "keras"
g_TrainSetDir = "data/modelTraining"
g_TestSetDir = "data/modelTesting"
g_ClassNames = None

g_EmotionImage = "data/exampleImages/neutral.jpg"

#-----------------------------------------CV
g_cvFaceClassifierName = "haarcascade_frontalface_default.xml"

def cvGetCascadeClassifier( name ):
    l_Classifier = cv2.CascadeClassifier( cv2.data.haarcascades + name )

    return l_Classifier

#-----------------------------------------MAIN_LOWER_FUNC
def CreateAndTrainNewModel():
    global g_Model, g_ClassNames

    g_Model = GetLearningModel();
    g_Model.summary()

    l_TrainItr = GetTrainingIterator( g_TrainSetDir, g_ReqImgSize, g_SetBatchSize )
    l_TestItr = GetTestingIterator( g_TestSetDir, g_ReqImgSize, g_SetBatchSize )

    l_History = TrainModel( g_Epochs, g_Model, l_TrainItr, l_TestItr )

    SaveModel( g_Model, g_ModelName )

    g_ClassNames = GetClassDict( l_TrainItr )

#-----------------------------------------MAIN_UPPER_FUNC
def InitSystem():
    global g_Model, g_ClassNames

    l_Path = GetFullModelPath( g_ModelsFolderName, g_ModelName, g_ModelExt )

    if not os.path.exists( l_Path ):
        print( "Creating Fresh Model\n" )
        CreateAndTrainNewModel()
    else:
        print( "Loading Existing Model\n" )
        g_Model = LoadModel( g_ModelName )
        g_Model.summary()
        g_ClassNames = GetFoldersDict( g_TrainSetDir )

    print( "\n\n\n" )


#-----------------------------------------MAIN

InitSystem()