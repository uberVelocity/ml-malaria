# Malaria image recognition
This project looks into Machine Learning approaches to detect Malaria in cells.

## Dependencies
TODO

## Dataset
The dataset can be found here: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria/data

When unpacking the dataset, only unpack the nested `cell_images/` directory. This directory contains two sets of data.
One directory (`Uninfected/`) contains images of normal cells, the other (`Parasitized/`) infected cell images.

## Configuring the project 
After making sure the data is available as specified above, users need to specify some global variables in the `config.py`
file. The file should contain at least the dataset's location on the machine as well as the size the images will be
scaled to.

After this is done, be sure to run the `preprocess_images.py` script before running any of the classifier scripts.

## Running the classifiers
The classifier scripts can be ran normally using `python3 name_of_script.py`.

## More info
Link to google docs: https://docs.google.com/document/d/1C4p8dU8fvSJrkm5Fhk6LJWS_y_pIMDuk0NQpQRRvteY/edit
