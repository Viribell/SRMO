from myModel import *
from myUtils import *
from myCV import *
from myTkinter import *
import matplotlib.pyplot as plt
import seaborn as seb
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

#Complete Dataset Source:
#https://www.kaggle.com/datasets/msambare/fer2013

g_Model = None

g_DropoutValue = 0.5
g_ClassNum = 7
g_ReqImgSize = (48, 48)
g_SetBatchSize = 64
g_Epochs = 170

g_ModelsFolderName = "models"
g_ModelName = "myModel_save_full_new"
g_ModelExt = "keras"
g_TrainSetDir = "data/modelTraining"
g_TestSetDir = "data/modelTesting"
g_ClassNames = None

g_EmotionImage = "data/exampleImages/neutral.jpg"
g_VideoCapture = None #!!!!!!!!!!!

g_tkWindow = None
g_tkWindowTitle = "Emotion Recognition"
g_tkWindowWidth = 1020
g_tkWindowHeight = 800
g_tkWindowDimension = str(g_tkWindowWidth) + "x" + str(g_tkWindowHeight)
g_tkBasicFont = ("Arial", 12)
g_tkTaskId = -1

#----
g_ImgLabel = None
g_ResultLabel = None
g_VideoLabel = None
g_VideoResultLabel = None
g_ImgPredictions = ""

g_CurrImgPath = None
g_CurrCroppedImg = None

g_StartedCapture = False
g_LastCapture = ""
g_LastFaces = []

#-----------------------------------------MAIN_LOWER_FUNC
def GetNormalisedEmotion( imgPath, imgSize ):
    global g_CurrCroppedImg

    l_Img = cvLoadImage( imgPath )
    l_Img = cvConvertImageToGrayscale( l_Img )

    l_Face = cvDetectOneByClassifier( l_Img, cvGetCascadeClassifier( g_cvFaceClassifierName ) )

    if len(l_Face) == 0:
        print( "Face not detected." )
        return

    l_Emotion = cvCropImgToArea( l_Img, l_Face, imgSize )
    g_CurrCroppedImg = l_Emotion

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

def PrintPredictions( predictions ):
    l_Out = ""    

    for i, prediction in enumerate( predictions[0] ):
        l_ClassName = GetFolderByValueFromDict( g_ClassNames, i )
        l_Normal = prediction * 100
        l_Out += f"{l_ClassName}: {l_Normal:.2f}%\n"

    return l_Out

def PrintPredictionsShort( predictions ):
    l_Out = ""
    l_Center = int(len( predictions[0] ) / 2)

    for i, prediction in enumerate( predictions[0] ):
        l_ClassName = GetFolderByValueFromDict( g_ClassNames, i )
        l_Normal = prediction * 100
        l_Out += f"[{l_ClassName[:2]}]{l_Normal:.2f}% "

        if i == l_Center: l_Out += "\n"

    return l_Out

def PrintModelLearningInfo( modelHistory ):
    with open( "trainHistory.pkl", "wb" ) as f:
        pickle.dump( modelHistory.history, f )

    l_AccHistory = modelHistory.history['accuracy']
    l_ValAccHistory = modelHistory.history['val_accuracy']
    l_LossHistory = modelHistory.history['loss']
    l_ValLossHistory = modelHistory.history['val_loss']
    l_Epochs = range(1, len(l_AccHistory) + 1)

    l_BestValAcc = max( l_ValAccHistory )
    l_BestValLoss = min( l_ValLossHistory )
    l_BestAccEpoch = l_ValAccHistory.index( l_BestValAcc ) + 1
    l_BestLossEpoch = l_ValLossHistory.index( l_BestValLoss ) + 1

    l_LastEpoch = len( l_AccHistory )
    l_LastAcc = l_AccHistory[-1]
    l_LastValAcc = l_ValAccHistory[-1]
    l_LastLoss = l_LossHistory[-1]
    l_LastValLoss = l_ValLossHistory[-1]

    print( f"---- Epoch {l_LastEpoch}/{l_LastEpoch} ----" )
    print( f"Train Accuracy: {l_LastAcc:.4f}" )
    print( f"Validation Accuracy: {l_LastValAcc:.4f}" )
    print( f"Train Loss: {l_LastLoss:.4f}" )
    print( f"Validation Loss: {l_LastValLoss:.4f}\n" )
    print( f"Best Validation Accuracy: {l_BestValAcc:.4f}, at Epoch {l_BestAccEpoch}/{l_LastEpoch}" )
    print( f"Best Validation Loss: {l_BestValLoss:.4f}, at Epoch {l_BestLossEpoch}/{l_LastEpoch}" )

    plt.figure( figsize=(12, 5) )

    plt.subplot( 1, 2, 1 )
    plt.plot( l_Epochs, l_AccHistory, 'b-', label='Training Acc' )
    plt.plot( l_Epochs, l_ValAccHistory, 'g-', label='Validation Acc' )
    plt.title( 'Training|Validation Acc' )
    plt.xlabel( 'Epochs' ); plt.ylabel( 'Accuracy' ); plt.legend(); plt.grid()

    plt.subplot( 1, 2, 2 )
    plt.plot( l_Epochs, l_LossHistory, 'b-', label='Training Loss' )
    plt.plot( l_Epochs, l_ValLossHistory, 'g-', label='Validation Loss' )
    plt.title( 'Training|Validation Loss' )
    plt.xlabel( 'Epochs' ); plt.ylabel( 'Loss' ); plt.legend(); plt.grid()

    plt.show()

    return

def CreateAndTrainNewModel():
    global g_Model, g_ClassNames

    g_Model = GetLearningModel();
    g_Model.summary()

    l_TrainItr = GetTrainingIterator( g_TrainSetDir, g_ReqImgSize, g_SetBatchSize )
    l_TestItr = GetTestingIterator( g_TestSetDir, g_ReqImgSize, g_SetBatchSize )

    l_Checkpoint = ModelCheckpoint(
        filepath = GetFullModelPath( g_ModelsFolderName, g_ModelName+"_best", g_ModelExt ),
        monitor = 'val_accuracy', save_best_only = True, mode = 'max', verbose = 1
    )

    l_History = TrainModel( g_Epochs, g_Model, l_TrainItr, l_TestItr, [l_Checkpoint] )

    PrintModelLearningInfo( l_History )
    SaveModel( g_Model, g_ModelName )

    g_ClassNames = GetClassDict( l_TrainItr )


def PrintModelInfo():
    l_TestItr = GetTestingIterator( g_TestSetDir, g_ReqImgSize, g_SetBatchSize )
    l_FullClasses = GetFullClasses( l_TestItr )
    l_ClassNames = list( g_ClassNames.keys() )
    l_TestLoss, l_TestAcc = g_Model.evaluate( l_TestItr )
    l_Predictions = g_Model.predict( l_TestItr )
    l_Predictions = np.argmax( l_Predictions, axis=1 )

    print( f"Test Accuracy: {l_TestAcc:.4f}" )
    print( f"Test Loss: {l_TestLoss:.4f}" )

    l_ConfusionMtx = confusion_matrix( l_FullClasses, l_Predictions )

    plt.figure(figsize=(8, 6))

    seb.heatmap(
        l_ConfusionMtx,
        annot=True, fmt='d', cmap='Oranges',
        xticklabels = l_ClassNames, yticklabels = l_ClassNames
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class"); plt.ylabel("True Class")

    plt.tight_layout()
    plt.show()

    print("True VS Predicted:\n")
    l_IsMatchedCounter = 0
    l_IsMatched = False
    l_TotalSamples = len( l_FullClasses )
    for i in range( l_TotalSamples ):
        l_TrueClasses = l_ClassNames[ l_FullClasses[i] ]
        l_PredictedClasses = l_ClassNames[ l_Predictions[i] ]
        l_IsMatched = l_TrueClasses == l_PredictedClasses
        
        if l_IsMatched: l_IsMatchedCounter += 1

        if i <= 15:
            print( f"{i}: True = {l_TrueClasses}, Predicted = {l_PredictedClasses}, Matches = {l_IsMatched}" )
        elif i == 16:
            print( "..." )
    print( f"\n Total Samples: {l_TotalSamples}, Correct Matches: {l_IsMatchedCounter}, \
        Incorrect Matches: {l_TotalSamples-l_IsMatchedCounter}" )

    l_ClassReport = classification_report(
        l_FullClasses, l_Predictions, target_names = l_ClassNames
    )

    print( "\nClassification Report:\n" )
    print( l_ClassReport )


def ClearResult():
    global g_ResultLabel, g_CurrCroppedImg

    g_ResultLabel.Image( None )
    g_ResultLabel.Text( "" )
    g_CurrCroppedImg = None


#-----------BUTTONS
def LoadImage():
    global g_ImgLabel, g_CurrImgPath

    l_Path = tkOpenFileDialog( "Images", "*.jpg *.jpeg *.png *.bmp" )

    if not l_Path:
        print( "File choice cancelled." )
        return

    if g_CurrCroppedImg is not None:
        ClearResult()

    g_CurrImgPath = l_Path

    l_Img = Image.open( l_Path )
    l_TKImg = cvImageToTKImage( l_Img )

    g_ImgLabel.Image( l_TKImg )

def DetectEmotion():
    global g_ResultLabel

    if not g_CurrImgPath: 
        print( "No filepath!" )
        return

    l_ClassName = ProcessImageForEmotion( g_CurrImgPath )

    if g_CurrCroppedImg is None:
        print( "No cropped img!" )
        return

    g_ResultLabel.Text( "Emotion: " + l_ClassName + "\n Predictions: \n" + g_ImgPredictions )
    g_ResultLabel.Image( cvCVImageToTKImage( g_CurrCroppedImg, g_ReqImgSize[0], g_ReqImgSize[1]) )

def UpdateVideoCapture():
    global g_VideoCapture, g_VideoLabel, g_VideoResultLabel, g_tkTaskId, g_LastCapture, g_ReqImgSize, g_LastFaces

    l_Ret, l_Frame = g_VideoCapture.read()
    if not l_Ret:
        g_VideoResultLabel.Text( "Error while trying to capture video." )
        g_VideoCapture = None
        return

    l_GrayCapture = cvConvertImageToGrayscale( l_Frame )
    l_Faces = cvDetectMultipleByClassifier( l_GrayCapture, cvGetCascadeClassifier( g_cvFaceClassifierName ), 1.2, 9 )

    if len(l_Faces) == 0: l_Faces = g_LastFaces
    else: g_LastFaces = l_Faces

    if len(l_Faces) != 0: 
        g_LastCapture = ""
        cvMarkDetectedAreas( l_Frame, l_Faces )

        for i, face in enumerate( l_Faces ):
            l_Emotion = cvCropImgToArea( l_GrayCapture, face, g_ReqImgSize )
            l_Emotion = cvNormaliseImg( l_Emotion )
            l_Emotion = cvExpandImgDimFromLeft( l_Emotion ) #batch
            l_Emotion = cvExpandImgDimFromRight( l_Emotion ) #channel
            l_Predictions = PredictEmotion( l_Emotion )
            l_ClassName = CategorizeEmotion( l_Predictions )
            l_ShortPredictions = PrintPredictionsShort( l_Predictions )

            g_LastCapture += f"Face {i}: {l_ClassName}, {l_ShortPredictions}\n"

    g_VideoLabel.Image( cvCVImageToTKImage( l_Frame ) )
    g_VideoResultLabel.Text( g_LastCapture )

    g_tkTaskId = tkScheduleTaskAfter( g_tkWindow, 10, UpdateVideoCapture )

def StartVideoCapture():
    global g_StartedCapture, g_VideoResultLabel, g_tkWindow, g_tkTaskId, g_VideoCapture

    if g_StartedCapture is True: return

    g_VideoCapture = cvGetDefaultVideoCapture()
    if not g_VideoCapture.isOpened():
        g_VideoCapture = None
        print( "Failed to find camera." )
        g_VideoResultLabel.Text( "Couldn't find any camera." )
        return

    g_StartedCapture = True

    g_tkTaskId = tkScheduleTaskAfter( g_tkWindow, 10, UpdateVideoCapture )



def StopVideoCapture():
    global g_StartedCapture, g_VideoLabel, g_tkWindow, g_tkTaskId, g_VideoCapture, g_LastFaces

    if g_StartedCapture is False: return

    tkCancelTask( g_tkWindow, g_tkTaskId )
    g_tkTaskId = -1
    g_StartedCapture = False

    g_VideoCapture.release()
    g_VideoCapture = None

    g_LastFaces = []


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
        PrintModelInfo()

    print( "\n\n\n" )

def InitWindow(): #!!!!!!!!!!!
    global g_tkWindow, g_ImgLabel, g_ResultLabel, g_VideoLabel, g_VideoResultLabel

    l_LeftFrameColor = "lightgray"
    l_RightFrameColor = "yellow"

    g_tkWindow = tkCreateWindow( g_tkWindowTitle, g_tkWindowDimension )

    l_LeftFrame = tkAddFrame( g_tkWindow, l_LeftFrameColor ).Dimension( g_tkWindowWidth / 2, 100 ).Pack( side="left", fill="y" ).PackPropagate( False )
    l_RightFrame = tkAddFrame( g_tkWindow, l_RightFrameColor ).Dimension( 100, 100 ).Pack( side="right", fill="both", expand=True ).PackPropagate( False )


    #----LEFT_FRAME
    tkAddLabel( l_LeftFrame.Get(), "Image Based Emotion Recognition" ).Pack( pady=8 ).Font( g_tkBasicFont ).Bg( l_LeftFrameColor )
    
    l_ButtonFrame = tkAddFrame( l_LeftFrame.Get() ).Pack( pady=8 )
    tkAddButton( l_ButtonFrame.Get(), "Load Image", LoadImage ).Pack( side="left", padx=8 ).Font( g_tkBasicFont )
    tkAddButton( l_ButtonFrame.Get(), "Detect Emotion", DetectEmotion ).Pack( side="left", padx=8 ).Font( g_tkBasicFont )

    l_ImgFrame = tkAddFrame( l_LeftFrame.Get() ).Pack( pady=8 )
    g_ImgLabel = tkAddLabel( l_ImgFrame.Get(), "" ).Pack( pady=8 ).Bg( l_LeftFrameColor )

    l_ResultFrame = tkAddFrame( l_LeftFrame.Get() ).Pack( pady=8 )
    g_ResultLabel = tkAddLabel( l_ResultFrame.Get(), "" ).Pack( pady=8 ).Bg( l_LeftFrameColor ).Config( compound="top" )
    #----LEFT_FRAME

    #----RIGHT_FRAME
    tkAddLabel( l_RightFrame.Get(), "Camera Based Emotion Recognition" ).Pack( pady=8 ).Font( g_tkBasicFont ).Bg( l_RightFrameColor )
    
    l_ButtonRightFrame = tkAddFrame( l_RightFrame.Get() ).Pack( pady=8 ).Bg( l_RightFrameColor )
    tkAddButton( l_ButtonRightFrame.Get(), "Start Capture", StartVideoCapture ).Pack( side="left", padx=8 ).Font( g_tkBasicFont )
    tkAddButton( l_ButtonRightFrame.Get(), "Stop Capture", StopVideoCapture ).Pack( side="left", padx=8 ).Font( g_tkBasicFont )

    l_VideoFrame = tkAddFrame( l_RightFrame.Get() ).Pack( pady=8 ).Bg( l_RightFrameColor )
    g_VideoLabel = tkAddLabel( l_VideoFrame.Get(), "" ).Pack( pady=8 ).Bg( l_RightFrameColor )

    l_ResultRightFrame = tkAddFrame( l_RightFrame.Get() ).Pack( pady=8 ).Bg( l_RightFrameColor )
    g_VideoResultLabel = tkAddLabel( l_ResultRightFrame.Get(), "" ).Pack( pady=8 ).Bg( l_RightFrameColor ).Config( compound="top" )

    #----RIGHT_FRAME

def ProcessImageForEmotion( imgPath ):
    global g_ImgPredictions

    g_ImgPredictions = ""
    l_Emotion = GetNormalisedEmotion( imgPath, g_ReqImgSize )
    l_Predictions = PredictEmotion( l_Emotion )
    l_ClassName = CategorizeEmotion( l_Predictions )

    g_ImgPredictions = PrintPredictions( l_Predictions )

    return l_ClassName

def HandleProgram():  #!!!!!!!!!!!
    global g_tkWindow

    g_tkWindow.mainloop()

    if g_VideoCapture is not None: g_VideoCapture.release()

#-----------------------------------------MAIN

InitSystem()
InitWindow()
HandleProgram()
