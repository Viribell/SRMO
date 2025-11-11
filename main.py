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
g_ClassNames = None

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

def GetTrainingDataGen():
    l_DataGen = ImageDataGenerator(
        rescale= 1./255,
        rotation_range= 30,
        shear_range= 0.3,
        zoom_range= 0.3,
        horizontal_flip= True,
        fill_mode = 'nearest'
    )

    return l_DataGen

def GetTestingDataGen():
    l_DataGen = ImageDataGenerator(
        rescale = 1./255
    )

    return l_DataGen

def GetTrainingIterator( dataDir, imgSize, batchSize ):
    l_DataGen = GetTrainingDataGen()

    l_TrainIterator = l_DataGen.flow_from_directory(
        dataDir,
        target_size = imgSize,
        color_mode = 'grayscale',
        batch_size = batchSize,
        class_mode = 'categorical',
        shuffle = True
    )

    return l_TrainIterator

def GetTestingIterator( dataDir, imgSize, batchSize ):
    l_DataGen = GetTestingDataGen()

    l_TestIterator = l_DataGen.flow_from_directory(
        dataDir,
        target_size = imgSize,
        color_mode = 'grayscale',
        batch_size = batchSize,
        class_mode = 'categorical',
        shuffle = False
    )

    return l_TestIterator

def GetClassDict( iterator ):
    return iterator.class_indices

def TrainModel( epochs, model, trainItr, testItr ):
    l_History = model.fit(
        trainItr,
        steps_per_epoch = len(trainItr),
        epochs = epochs,
        validation_data = testItr,
        validation_steps = len(testItr)
    )

    return l_History

def SaveModel( model, fileName, mainFolder="models", modelExt="keras" ):
    l_Path = GetFullModelPath( mainFolder, fileName, modelExt )

    os.makedirs( mainFolder, exist_ok=True )
    model.save( l_Path )

def LoadModel( fileName, mainFolder="models", modelExt="keras" ):
    l_Path = GetFullModelPath( mainFolder, fileName, modelExt )

    if not os.path.exists( l_Path ):
        print( "Model file doesn't exist!" )
        return False

    l_Model = load_model( l_Path )  

    return l_Model

#-----------------------------------------UTILS
def GetFoldersDict( datasetPath ):
    l_Folders = []
    l_FolderIndices = {}

    for item in os.listdir( datasetPath ):
        itemPath = os.path.join( datasetPath, item )

        if os.path.isdir( itemPath ): l_Folders.append( item )

    l_Folders.sort()

    for index, folder in enumerate( l_Folders ):
        l_FolderIndices[folder] = index

    return l_FolderIndices

def GetFolderByValueFromDict( folderDict, index ):
    l_Key = None

    for key, value in folderDict.items():
        if value == index: l_Key = key

    return l_Key

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