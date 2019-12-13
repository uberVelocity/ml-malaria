import cv2
import os
import numpy as np

if __name__ == "__main__":
    dataset = os.environ["HOME"] + "/dataset"
    folders = os.listdir(dataset)
    for folder in folders:
        folderName = folder
        folder = os.path.join(dataset,folder)
        if os.path.isdir(folder):
            files = os.listdir(folder)
            imgList = []
            saveFolderPath = os.path.join(folder,"Cropped")
            for fileName in files:
                fileAdress = os.path.join(folder,fileName)
                if not os.path.exists(saveFolderPath):
                    os.mkdir(saveFolderPath)
                if os.path.isfile(fileAdress) and os.path.splitext(fileName)[-1].lower() == ".jpg":
                    imgFile = fileAdress
                    saveAdress = saveFolderPath + "/" + folderName+ "-" + os.path.splitext(fileName)[0]
                    imgNpy = folder + "/" + os.path.splitext(fileName)[0] + ".npy"
                    image = cv2.imread(imgFile, cv2.IMREAD_COLOR)
                    npyFile = np.load(imgNpy)
                    top = int(npyFile[0])
                    bottom = int(npyFile[1])
                    left = int(npyFile[2])
                    right = int(npyFile[3])
                    image_crop = image[top:bottom,left:right]
                    image_crop = cv2.resize(image_crop, (220,180), interpolation=cv2.INTER_AREA)
                    # print(image_crop)
                    # image_crop = cv2.normalize(image_crop, image_crop, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

                    brightened_image = cv2.convertScaleAbs(image_crop, alpha=1.5, beta=0)
                    
                    darkened_image = cv2.convertScaleAbs(image_crop, alpha=0.9, beta=0)
                    
                    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
                    hsv[...,1] = hsv[...,1] * 2.0
                    over_saturated_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                   
                    hsv[...,1] = hsv[...,1] / 8.0
                    under_saturated_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                    
                    image_crop = np.true_divide(image_crop, 255)
                    cv2.imwrite(saveAdress + '-cropped.jpg', image_crop)
                    brightened_image = np.true_divide(brightened_image, 255)
                    cv2.imwrite(saveAdress + '-brightened.jpg', brightened_image)
                    darkened_image = np.true_divide(darkened_image, 255)
                    cv2.imwrite(saveAdress + '-darkened.jpg', darkened_image)
                    over_saturated_image = np.true_divide(over_saturated_image, 255)
                    cv2.imwrite(saveAdress + '-oversat.jpg', over_saturated_image)
                    under_saturated_image = np.true_divide(under_saturated_image, 255)
                    cv2.imwrite(saveAdress + '-undersat.jpg', under_saturated_image)
                    imgList.append(image_crop)
                    imgList.append(brightened_image)
                    imgList.append(darkened_image)
                    imgList.append(over_saturated_image)
                    imgList.append(under_saturated_image)
                    #cv2.waitKey(0)

            np.save(saveFolderPath + '/' + folderName + '-data.npy',imgList)
                    