import numpy as np
import argparse
import pickle
import cv2
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



# parametry przy uruchamianiu
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
ap.add_argument('-m', '--model', required=False, help='path to model', default='output\model_custom_VGG16_2_classes.hdf5')
ap.add_argument('-l', '--labels', required=False, help='path to labels', default='output\labels_2_classes.pickle')
args = vars(ap.parse_args())


INPUT_SHAPE = (224, 224, 3)



print('[INFO] Wczytywanie modelu.')
model = load_model(args['model'])
image = args['image']
labels = args['labels']


def predict_class(image_path, model, input_shape, labels_path):
    '''
    Funkcja do klasyfikacji pojedynczego zdjęcia.
    Na wyjściu pokazuje zdjęcie z etykietą i przewdopodobieństwem.
    '''
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = image.astype('float') / 255.
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)


    y_pred = model.predict(image)[0]

    # wczytanie etykiet
    with open(labels_path, 'rb') as file:
        classes = pickle.loads(file.read())

    labels = dict(enumerate(classes))

    print(f'y_pred: {y_pred}')
    pred_class = np.argmax(y_pred)
    text = f'Label: {labels[pred_class]}, probability: {100 * y_pred[pred_class]:.2f}%'
    print(text)

    # załadowanie obrazu
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    # wyświetlenie obrazu z etykietą i prawdopodobieństwem
    cv2.putText(img=image, text=text,
                org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 255, 0), thickness=2)

    cv2.imshow('my image', image)
    cv2.waitKey(0)



predict_class(image, model, INPUT_SHAPE, labels)


'''
Przykład uruchomienia w terminalu:
>>>> python 06_predict_class.py -i example_images/ok_01.jpg
'''