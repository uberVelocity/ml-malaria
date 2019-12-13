import cv2
import os
import numpy as np

if __name__ == "__main__":
    dataset = os.environ["HOME"] + "/dataset"
    folders = os.listdir(dataset)
    for folder in folders:
        folderName = folder
        folder = os.path.join(dataset, folder)
        imgList = []
        if os.path.isdir(folder):
            saveFolder = os.path.join(folder,"Cropped")
            images = os.listdir(saveFolder)
            for img in images:
                imgAdr = os.path.join(saveFolder,img)
                imgRead = cv2.imread(imgAdr, cv2.IMREAD_COLOR)
                imgList.append(imgRead)
                cv2.imshow("window",imgRead)
                cv2.waitKey(0)
            np.save(saveFolder + '/' + folderName + '-data.npy',imgList)