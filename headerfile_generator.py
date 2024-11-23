import os
import numpy as np

if __name__ == '__main__':
    # giving directory name
    dirname = "C:\\Users\\Broga\\Desktop\\ba-misc\\novelty-detection-analysis-system\\NDAS\\testing\\bed2013"

    # set the dir for the resulting header file
    toDirname = "C:\\Users\\Broga\\Desktop\\ba-misc\\masterheaderfile.hea"
    
    # setting naming scheme of headerfiles to be read
    # if you want to read only one specific file, set namingScheme to the name of the file
    namingScheme = 'UnoViS_bed2013_'
    
    # giving file extension
    ext = ('.hea')
    
    # setting the number of displayed files. If set to -1 will display all files
    numberOfDisplayedFiles = -1
 
    # iterating over all files
    files = os.listdir(dirname)
    
    numberOfFiles = 0
    overallFileSize = 0
    fileNames = []
    fileSizes = []

    for file in files:
        
        # only iterate over files with the desired naming scheme
        if file.startswith(namingScheme):
            # checking file extension
            if file.endswith(ext):
                fileNames.append(file)
                numberOfFiles += 1
                print(file) # printing file name of desired extension
                
                with open(os.path.join(dirname, file), 'r') as f:
                    contents = f.readlines()
                    numberOfChanels = contents[0].split(' ')[1]
                    readoutFrequency = contents[0].split(' ')[2]
                    fileSizes.append(contents[0].split(' ')[3].removesuffix('\n'))
                    channelNames = []

                    for line in contents[1:]:
                        if line.startswith('#'):
                            continue
                        else:
                            channelNames.append(line.split(' ')[-1].removesuffix('\n'))
            else:
                continue
        
    fileInformation = np.vstack((fileNames, fileSizes)).T
    
    if numberOfDisplayedFiles == -1:
        for line in fileInformation:
            if(line[0].startswith(namingScheme)):
                overallFileSize += int(line[1])
    else:
        for line in fileInformation[:numberOfDisplayedFiles]:
            if(line[0].startswith(namingScheme)):
                overallFileSize += int(line[1])
        
    
    file = open(toDirname, 'w+')
    file.write('UnoViS_bed2013/' + str(numberOfFiles)+ " " + str(numberOfChanels) + " " + str(readoutFrequency) + " " + str(overallFileSize) + '\n')
    
    if numberOfDisplayedFiles == -1:
        for line in fileInformation:
            if(line[0].startswith(namingScheme)):
                file.write(line[0] + " " + line[1] + '\n')
    else:
        for line in fileInformation[:numberOfDisplayedFiles]:
            if(line[0].startswith(namingScheme)):
                file.write(line[0] + " " + line[1] + '\n')
                
    print('----------Process Successful----------')
    print('headerfile name' , 'size of .dat file that headerfile is referring to')
    print(fileInformation)
    print('Readout frequency: ', readoutFrequency)
    print('Number of channels: ', numberOfChanels)
    print('Channel names: ', channelNames)
    print('Number of files: ', numberOfFiles)
    print('Number of displayed files: ',  'All' if numberOfDisplayedFiles == -1 else numberOfDisplayedFiles)