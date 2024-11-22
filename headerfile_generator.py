import os
import numpy as np

if __name__ == '__main__':
    # giving directory name
    dirname = "C:\\Users\\Broga\\OneDrive\\Desktop\\ba-misc\\bed2013"

    # giving file extension
    ext = ('.hea')
 
    # iterating over all files
    files = os.listdir(dirname)
    numberOfFiles = 0
    fileNames = []
    fileSizes = []

    for file in files:
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
    print(fileInformation)
    print('Readout frequency: ', readoutFrequency)
    print('Number of channels: ', numberOfChanels)
    print('Channel names: ', channelNames)
    print('Number of files: ', numberOfFiles)
    print('File names: ', fileNames)
    


