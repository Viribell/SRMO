from myModel import *
from myUtils import *
from myCV import *
from myTkinter import *

#Complete Dataset Source:
#https://www.kaggle.com/datasets/msambare/fer2013

g_Model = None

g_DropoutValue = 0.5
g_ClassNum = 7
g_ReqImgSize = (48, 48)
g_SetBatchSize = 64
g_Epochs = 30

g_ModelsFolderName = "models"
g_ModelName = "myModel_save_full"
g_ModelExt = "keras"
g_TrainSetDir = "data/modelTraining"
g_TestSetDir = "data/modelTesting"
g_ClassNames = None

g_EmotionImage = "data/exampleImages/neutral.jpg"

g_tkWindow = None
g_tkWindowTitle = "Emotion Recognition"
g_tkWindowWidth = 1020
g_tkWindowHeight = 800
g_tkWindowDimension = str(g_tkWindowWidth) + "x" + str(g_tkWindowHeight)
g_tkBasicFont = ("Arial", 12)


#----
g_ImgLabel = None
g_ResultLabel = None

#-----------------------------------------MAIN_LOWER_FUNC
def GetNormalisedEmotion( imgPath, imgSize ):
    l_Img = cvLoadImage( imgPath )
    l_Img = cvConvertImageToGrayscale( l_Img )

    l_Face = cvDetectOneByClassifier( l_Img, cvGetCascadeClassifier( g_cvFaceClassifierName ) )

    if len(l_Face) == 0:
        print( "Face not detected." )
        return

    l_Emotion = cvCropImgToArea( l_Img, l_Face, imgSize )
    l_Emotion = cvNormaliseImg( l_Emotion )
    l_Emotion = cvExpandImgDimFromLeft( l_Emotion ) #batch
    l_Emotion = cvExpandImgDimFromRight( l_Emotion ) #channel

    return l_Emotion

def PredictEmotion( emotionImg ):
    l_Predictions = g_Model.predict( emotionImg )

    return l_Predictions

def CategorizeEmotion( predictions ):
    l_ClassIndex = np.argmax( predictions )
    l_ClassName = GetFolderByValueFromDict( g_ClassNames, l_ClassIndex )

    return l_ClassName

def CreateAndTrainNewModel():
    global g_Model, g_ClassNames

    g_Model = GetLearningModel();
    g_Model.summary()

    l_TrainItr = GetTrainingIterator( g_TrainSetDir, g_ReqImgSize, g_SetBatchSize )
    l_TestItr = GetTestingIterator( g_TestSetDir, g_ReqImgSize, g_SetBatchSize )

    l_History = TrainModel( g_Epochs, g_Model, l_TrainItr, l_TestItr )

    SaveModel( g_Model, g_ModelName )

    g_ClassNames = GetClassDict( l_TrainItr )

def Test():
    print()
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

def InitWindow():
    global g_tkWindow, g_ImgLabel, g_ResultLabel

    l_LeftFrameColor = "lightgray"
    l_RightFrameColor = "yellow"

    g_tkWindow = tkCreateWindow( g_tkWindowTitle, g_tkWindowDimension )

    l_LeftFrame = tkAddFrame( g_tkWindow, l_LeftFrameColor ).Dimension( g_tkWindowWidth / 2, 100 ).Pack( side="left", fill="y" ).PackPropagate( False )
    l_RightFrame = tkAddFrame( g_tkWindow, l_RightFrameColor ).Dimension( 100, 100 ).Pack( side="right", fill="both", expand=True ).PackPropagate( False )


    #----LEFT_FRAME
    tkAddLabel( l_LeftFrame.Get(), "Image Based Emotion Recognition" ).Pack( pady=8 ).Font( g_tkBasicFont ).Bg( l_LeftFrameColor )
    
    l_ButtonFrame = tkAddFrame( l_LeftFrame.Get() ).Pack( pady=8 )
    tkAddButton( l_ButtonFrame.Get(), "Load Image", Test ).Pack( side="left", padx=8 ).Font( g_tkBasicFont )
    tkAddButton( l_ButtonFrame.Get(), "Detect Emotion", Test ).Pack( side="left", padx=8 ).Font( g_tkBasicFont )

    l_ImgFrame = tkAddFrame( l_LeftFrame.Get() ).Pack( pady=8 )
    g_ImgLabel = tkAddLabel( l_ImgFrame.Get(), "" ).Pack( pady=8 ).Bg( l_LeftFrameColor )

    l_ResultFrame = tkAddFrame( l_LeftFrame.Get() ).Pack( pady=8 )
    g_ResultLabel = tkAddLabel( l_ResultFrame.Get(), "" ).Pack( pady=8 ).Bg( l_LeftFrameColor ).Config( compound="top" )
    #----LEFT_FRAME

    #----RIGHT_FRAME
    tkAddLabel( l_RightFrame.Get(), "Camera Based Emotion Recognition" ).Pack( pady=8 ).Font( g_tkBasicFont ).Bg( l_RightFrameColor )
    #----RIGHT_FRAME

def ProcessImageForEmotion( imgPath ):
    l_Emotion = GetNormalisedEmotion( imgPath, g_ReqImgSize )
    l_Predictions = PredictEmotion( l_Emotion )
    l_ClassName = CategorizeEmotion( l_Predictions )

    return l_ClassName

def HandleProgram():

    g_tkWindow.mainloop()


#-----------------------------------------MAIN

InitSystem()
InitWindow()
HandleProgram()