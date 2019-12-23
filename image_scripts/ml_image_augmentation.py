import cv2
import os
import numpy as np

if __name__ == "__main__":
    dataset = os.environ["HOME"] + "/Documents/University/ml/cell-images-for-detecting-malaria/cell_images/cell_images"
    folders = os.listdir(dataset)
    for folder in folders:
        folder_name = folder
        folder = os.path.join(dataset,folder)
        if os.path.isdir(folder):
            files = os.listdir(folder)
            img_list = []
            save_folder_path = os.path.join(folder, "augmented")
            for file_name in files:
                file_address = os.path.join(folder, file_name)
                if not os.path.exists(save_folder_path):
                    os.mkdir(save_folder_path)
                if os.path.isfile(file_address) and os.path.splitext(file_name)[-1].lower() == ".png":
                    img_file = file_address
                    save_address = save_folder_path + "/" + folder_name+ "-" + os.path.splitext(file_name)[0]
                    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
                    image_crop = cv2.resize(image, (80, 50), interpolation=cv2.INTER_AREA)

                    image = np.true_divide(image_crop, 255)

                    cv2.imwrite(save_address + 'normalized.png', image)
                    img_list.append(image)

            np.save(save_folder_path + '/' + folder_name + '-data.npy',img_list)
            print('saved ', save_folder_path)      