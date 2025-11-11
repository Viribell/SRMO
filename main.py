import os
import numpy as np

os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

g_Model = None

g_ModelsFolderName = "models"
g_ModelName = "myModel_save_full"
g_ModelExt = "keras"
g_TrainSetDir = "data/modelTraining"
g_TestSetDir = "data/modelTesting"

#-----------------------------------------MODEL
def GetLearningModel( dropoutValue=0.5, classNumber=7, inputShape=(48,48,1), filterSize=(3,3) ):

    l_Model = Sequential([
        Input( shape = inputShape ),

        Conv2D( 32, filterSize, activation='relu' ),
        MaxPooling2D( 2,2 ),

        Conv2D( 64, filterSize, activation='relu' ),
        MaxPooling2D( 2,2 ),

        Conv2D( 128, filterSize, activation='relu' ),
        MaxPooling2D( 2,2 ),

        Flatten(),
        Dense( 128, activation='relu' ),
        Dropout( dropoutValue ),
        Dense( classNumber, activation='softmax' )
    ])

    l_Model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )

    return l_Model

def GetFullModelPath( mainFolder, modelName, modelExt ):
    fullPath = mainFolder + "/" + modelName + "." + modelExt

    return fullPath

#-----------------------------------------MAIN_LOWER_FUNC
def CreateAndTrainNewModel():
    global g_Model

    g_Model = GetLearningModel();
    g_Model.summary()


#-----------------------------------------MAIN_UPPER_FUNC
def InitSystem():
    global g_Model

    l_Path = GetFullModelPath( g_ModelsFolderName, g_ModelName, g_ModelExt )

    if not os.path.exists( l_Path ):
        print( "Creating Fresh Model\n" )
        CreateAndTrainNewModel()
    else:
        print( "Loading Existing Model\n" )

    print( "\n\n\n" )


#-----------------------------------------MAIN

InitSystem()