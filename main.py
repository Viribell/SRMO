from myModel import *

import os


g_Model = None

g_ModelsFolderName = "models"
g_ModelName = "myModel_save_full"
g_ModelExt = "keras"
g_TrainSetDir = "data/modelTraining"
g_TestSetDir = "data/modelTesting"
g_ClassNames = None

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