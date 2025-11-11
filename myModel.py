import os
import numpy as np

os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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