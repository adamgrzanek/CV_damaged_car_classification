import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


print(tf.__version__) # 2.15.0


# parametry przy uruchamianiu
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to image dataset')
ap.add_argument('-m', '--model', required=False, help='path to model', default='output/model_custom_VGG16_2_classes.hdf5')
args = vars(ap.parse_args())
print(f'Podane argumenty: {args}\n') # {'dataset': 'test_images', 'model': 'output/model_custom_VGG16_2_classes.hdf5'}


INPUT_SHAPE = (224, 224, 3)


# tworzymy generator
datagen = ImageDataGenerator(
    rescale=1. / 255.
)

generator = datagen.flow_from_directory(
    directory=args['dataset'],
    target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)




print('[INFO] Wczytywanie modelu.')
model = load_model(args['model'])
print(model.summary())



# metoda predict
y_prob = model.predict(generator, workers=1)

# odczytanie klasy
pred_classes = [np.argmax(pred) for pred in y_prob]

# wyciągnięcie prawdopodobieństwa
prob = [np.round(100 * pred[np.argmax(pred)], 2) for pred in y_prob]

# pobranie prawdziwych klas
y_true = generator.classes

# df z wynikami
predictions = pd.DataFrame({'prob': prob, 'y_pred': pred_classes, 'y_true': y_true},
                           index=generator.filenames)  # indeks - nazwa pliku
predictions['is_incorrect'] = (predictions['y_true'] != predictions['y_pred']) * 1  # czy źle sklasyfikowany


print('\n====')
label_map = generator.class_indices
print(label_map)  # {'damaged_cars': 0, 'ok_cars': 1}
print('====')

# dodanie etykiet do df
label_map = dict((v, k) for k, v in label_map.items())
predictions['class'] = predictions['y_pred'].apply(lambda x: label_map[x])  # nazwa klasy

# fragment df ze źle sklasyfikowanymi obrazami
errors = predictions[predictions['is_incorrect'] == 1]['prob']

# przydzielone klasy do np.array([0, 0, 1,...,0]) do wyświetlenia raportów
y_pred = predictions['y_pred'].values

print(f'\n[INFO] Macierz konfuzji:\n{confusion_matrix(y_true, y_pred)}')
print(f'\n[INFO] Raport klasyfikacji:\n{classification_report(y_true, y_pred, target_names=generator.class_indices.keys())}')
print(f'\n[INFO] Dokładność modelu: {accuracy_score(y_true, y_pred) * 100:.2f}%')

# df to csv
predictions.to_csv(r'output/predictions.csv')

print(f'\n[INFO] Błędnie sklasyfikowano: {len(errors)}')
print('\n[INFO] Nazwy plików:')
print(errors)



'''
Przykład uruchomienia w terminalu:
>>>> python 05_check_model.py -d test_images 
'''