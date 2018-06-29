# Ethnicity Classifier

## Training
python train.py 
python train_faces.py (model on extracted faces)
In prediction, the models are combined, if the algorithm manage to extract faces, it uses the face model, 
otherwise it predicts with the other.

## Model download
Download the models [here](https://drive.google.com/open?id=1xrT7Nn-ErWDEZrq1Pt109odfBpcAnCHd) and put them in a directory called models.

## Predictions
Insert images in test_data and run python predict.py to classify the images
The model used for the prediction is the VGG16 pretrained on Imagenet.

* python predict.py --data\_dir '/test/images/folder/'

Now enjoy your ethnicity classifier!
