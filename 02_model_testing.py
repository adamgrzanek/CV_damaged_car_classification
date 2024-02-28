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

import numpy as np
import tensorflow as tf
import cv2 # do pracy ze zdjęciami
from tensorflow.keras.preprocessing.image import img_to_array # konwertuje obraz do tablicy
import os

# suppress logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.keras.optimizers import Adam  # optymalizator
from architecture import models  # import naszego modułu ze stworzonymi klasami modeli (models.py w katalogu architecture)

input_shape = (150, 150, 3)


# wczytanie modelu do testu
# !!!! do HEATMAP potrzebne 2 klasy więć tu w ostatniej warstwie 2 neurony i softmax!!!!
#model = models.LeNet5(input_shape=input_shape, num_classes=2, final_activation='softmax')
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

    array_img = img_to_array(image_resized)  # zamiana na tablice numpy
    expanded_sample_image = np.expand_dims(array_img, axis=0)

    # wyswietlenie informacji do sprawdzenia
    print(array_img.shape) # (150, 150, 3)
    print(array_img.reshape(1, -1).shape) # (1, 67500)  = flatten()
    print(expanded_sample_image.shape) # (1, 150, 150, 3)





print('')
print('\nPredykcja')
pred = model.predict(expanded_sample_image)
print(pred)




print('\nKompilacja modelu')
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy', # bo tu do heatmapy potrzeba 2 klas
              metrics=['accuracy'])




print('\nEwaluacja')
model.evaluate(expanded_sample_image, np.array([[0, 1]]))
