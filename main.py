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

def cvLoadImage( imgPath ):
    l_Img = cv2.imread( imgPath ) #in BGR scale by default

    return l_Img

def cvConvertImageToGrayscale( img ):
    l_GrayImg = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

    return l_GrayImg

def cvResizeImage( img, imgSize ):
    l_Img = cv2.resize( img, imgSize )

    return l_Img

def cvDetectOneByClassifier( img, classifier, imgScale=1.2, minNeighbours=5 ):
    if not isinstance( classifier, cv2.CascadeClassifier ):
        raise TypeError( "Given classifier is not CascadeClassifier!" )
  
    l_DetectedArea = classifier.detectMultiScale( img, imgScale, minNeighbours )[0]

    return l_DetectedArea

def cvCropImgToArea( img, detectedArea, imgSize ):
    l_AreaX, l_AreaY, l_AreaWidth, l_AreaHeight = detectedArea

    l_CroppedImg = img[ l_AreaY:l_AreaY + l_AreaHeight, l_AreaX:l_AreaX + l_AreaWidth ]
    l_CroppedImg = cvResizeImage( l_CroppedImg, imgSize )

    return l_CroppedImg

def cvNormaliseImg( img ):
    l_NormalisedImg = img.astype( "float" ) / 255.0 #normalise

    return l_NormalisedImg

def cvExpandImgDimFromLeft( img ):
    l_ExpandedImg = np.expand_dims( img, axis=0 )

    return l_ExpandedImg

def cvExpandImgDimFromRight( img ):
    l_ExpandedImg = np.expand_dims( img, axis=-1 )

    return l_ExpandedImg

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