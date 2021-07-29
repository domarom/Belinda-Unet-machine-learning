#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os #conda
import random #conda
import numpy as np #pip
from tqdm import tqdm #pip
from skimage.io import imread, imshow #pip install scikit-image
from skimage.transform import resize
import matplotlib.pyplot as plt #pip matplotlib 
import glob #pip install glob2
import cv2 #pip install opencv-python
from matplotlib import pyplot as plt


# In[ ]:


test_predictions = np.load('test_set_dice_inverted_masks-TL6.npy')


# In[ ]:


test_predictions.shape #shape of test set 


# In[ ]:


plt.imshow(np.squeeze(test_predictions[10]))
plt.colorbar()


# In[ ]:


roi = []
image_area = test_predictions.shape[1] * test_predictions.shape[2]


# In[ ]:


from scipy.misc 
import toimage, imsave
import imageio


# In[ ]:


import imageio
for i in range(316):
    imsave("D:/predicted_masks/TL6/slice_{0}.png".format(i), img_array[i,...])


# In[ ]:


for i in range(test_predictions.shape[0]):
    img = test_predictions[i]
    no_of_bg_pixels = cv2.countNonZero(img)
    area_ratio = ( no_of_bg_pixels)
    roi.append(area_ratio)


# In[ ]:


plt.plot(roi)
plt.xlabel("Time")
plt.ylabel("Area")
plt.title("TL5-Area of interest vs time")
plt.savefig("D:/Belinda/AreaVsTime.pdf")
plt.show()


# In[ ]:


data = np.array(roi)


# In[ ]:



# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
data = data
# save to csv file
savetxt('D:/Belinda/TL5-area-recropped.csv', data, delimiter=',')


# In[ ]:


np.where(temp <0.05)


# In[ ]:


plt.imshow(np.squeeze(test_predictions[816]))
plt.colorbar()


# In[ ]:




