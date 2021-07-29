# Belinda-Unet-machine-learning
Sea Sponge classification with complex background using a modified version of Unet, a CNN architecture 

Original article published: 

Harrison, D., De Leo Cabreara, F., Marini, S., Gallin, W., Mir, F., Leys, S. (2021) Using machine learning to recognize and map sponge behavior associated with multivariate climate parameters presented at the following, Water: Special issue: Pattern Analysis, Recognition and Classification of Marine Data. Submitted July, 2021.

Abstract:

Biological data sets are increasingly becoming more dense and complex in information. As big data is becoming the norm, the necessary link between biology and computer science needs to be made. We use a case of study of classifying sponge behavior to demonstrate the applicability of Convolution Neural Network (CNN) and to provide a roadmap to the specific CNN architecture, Unet. We have analyzed a large time series of hourly high-definition still images between 2012-2015 focusing on behavioral responses by an individual marine sponge, Suberites concinnus (Demosponge, Suberitidae). The multi-year, time-series was collected off the East coast of Vancouver Island Canada using the NEPTUNE seafloor cabled observatory infrastructure. We successfully performed semantic segmentation using the Unet architecture (for a Convolutional Neural Network) with some modifications. We adapted parts of the architecture to be more applicable to three channeled imagery (RGB) to achieve very successful losses, accuracies and dice scores; the best results were  0.03, 0.98 and 0.97 respectively. Some alterations that made this model so successful were the use of a dice-loss coefficient, adam optimizer and a dropout function after each convolutional layer. The model was validated with a five cross-fold validation. This investigation is the first step in understanding and modeling the behavior of a demosponge in a coastal environment setting, often subject to severe seasonal and inter-annual changes related to climate. This investigation also provides a roadmap to other experts in the field that are seeking to cross the interdisciplinary boundaries between  biology and computer science. 

Instructions:
1. **Datasets**
The model we generated would work well for biological questions investigating datasets of time series, with unique background and foreground image types, motion detection (behavior mapping), automated identification of individuals or specific identifiable parameters, classification of groups of organisms to name a few. 



2. **Ground truthing**
Use some type of image labeller to classify all pixel in the image. This can be done in python. However, we used matlab's image labeller: https://www.mathworks.com/help/vision/ug/get-started-with-the-image-labeler.html. Do this for between 30 to 60% of the data set. Our initial masks/labelled images we used 40% of the images.

3. **Running the model**
There are three sets of code in this repository: Dice_loss, cross-validation, and Timeseries plot. For each code there are two file types : jupyter notebook (.IPYNB) and python (.py).


4. **About the code published here**
