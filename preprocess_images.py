import cv2
import os
import numpy as np
import config

"""
Collects images from folders and produces normalized dataset as .npy files.
"""
if __name__ == "__main__":
    dataset_path = os.environ["HOME"] + config.image_location
    folders = os.listdir(dataset_path)

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            img_list = []

            save_folder_path = os.path.join(folder_path, "augmented")
            if not os.path.exists(save_folder_path):
                os.mkdir(save_folder_path)

            for file_name in files:
                file_address = os.path.join(folder_path, file_name)

                if os.path.isfile(file_address) and os.path.splitext(file_name)[-1].lower() == ".png":
                    save_address = save_folder_path + "/" + folder + "-" + os.path.splitext(file_name)[0]

                    # preprocess the image
                    image = cv2.imread(file_address, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (config.resize_to[0], config.resize_to[1]), interpolation=cv2.INTER_AREA)
                    image = np.true_divide(image, 255)

                    cv2.imwrite(save_address + 'normalized.png', image)
                    img_list.append(image)

            np.save(save_folder_path + '/' + folder + '-data.npy', img_list)
            print('saved ', save_folder_path)
