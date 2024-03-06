import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os

# suppress logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.optimizers import Adam  # optymalizator
from architecture import models

'''
Algorytm postępowania:
1. Napisać model
2. Wykonać predykcję na losowym tensorze
3. Skompiluje model
4. Wykonam ewaluacje na losowych danych lub probce wlasciwych danych
5. Wykonac trening na losowych danych
6. Przygotowac dane do konca
7. Powtorzyc kroki 2. - 5. na wlasciwych danych
'''

input_shape = (150, 150, 3)

# wczytanie modelu do testu
# do HEATMAP potrzebne 2 klasy więc tu w ostatniej warstwie 2 neurony i funkcja aktywacji softmax
model = models.custom_VGG16(input_shape=input_shape, num_classes=2, final_activation='softmax')
model = model.build()

# podsumowanie modelu
print(model.summary())




if __name__ == '__main__':
    sample_image_path = r'I:\PycharmProjects\CV_damaged_car_classification\cars_image\ok_cars\0001.jpg'

    # przygotowanie przykładowego zdjęcia
    # wczytanie zdjęcia
    image = cv2.imread(sample_image_path)
    cv2.imshow('original_image', image)

    # zmiana wymiaru
    image_resized = cv2.resize(image, (input_shape[0], input_shape[1]))
    cv2.imshow('resized_image', image_resized)
    cv2.waitKey(0)

    # zamiana na tablice numpy
    array_img = img_to_array(image_resized)
    expanded_sample_image = np.expand_dims(array_img, axis=0)

    # wyswietlenie informacji do sprawdzenia
    print(array_img.shape)
    print(array_img.reshape(1, -1).shape)
    print(expanded_sample_image.shape)

print('\nPredykcja')
pred = model.predict(expanded_sample_image)
print(pred)

print('\nKompilacja modelu')
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('\nEwaluacja')
model.evaluate(expanded_sample_image, np.array([[0, 1]]))
