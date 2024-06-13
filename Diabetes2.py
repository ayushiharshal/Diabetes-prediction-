# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:47:41 2024

@author: KIIT
"""

import numpy as np
import pickle


# loading the saved model
loaded_model = pickle.load(open('C:/Users/KIIT/Desktop/Disease Web app/Diabetes/trained_model.sav', 'rb'))

input_data = (6,148,72,35,0,33.6,0.627,50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
