import os

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