import numpy as np
import pandas as pd
from datetime import datetime    # do zapisania jako znacznik daty i czasu
import argparse    # do przekazywania argumentów podczas uruchamiania skryptu
import pickle      # do zachowania pewnych objektów, które stworzymy po drodze (słownik z mapowaniem etykiet)
import os
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # do augumentacji danych
from tensorflow.keras.callbacks import ModelCheckpoint  # do zapisania najlepszego modelu podczas treniwania
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam  # optymalizator
from architecture import models  # import naszego modułu (models.py) (ze stworzonymi klasami modeli)
import warnings # do ograniczenia printowania logów (dzienników)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



print(f'Tensorflow version: {tf.__version__}')


# argumenty przy odpalaniu z termianala: liczba epok
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=1, help='Określ liczbę epok', type=int)
args = vars(ap.parse_args()) # -> {'epochs': 1}


# stałe
LEARNING_RATE = 0.001
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
TRAIN_DIR = 'splitted_images/train'
VALID_DIR = 'splitted_images/valid'


# wykres historii uczenia
def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines', marker_color='#f29407'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines', marker_color='#0771f2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines', marker_color='#f29407'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines', marker_color='#0771f2'), row=2, col=1)

    fig.update_xaxes(title_text='Liczba epok', row=1, col=1)
    fig.update_xaxes(title_text='Liczba epok', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f"Metrics: {MODEL_NAME}")

    po.plot(fig, filename=filename, auto_open=False)


# generator zbioru treningowego
train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1. / 255.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# generator zbioru walidacyjnego (tylko zmiana rozmiaru)
valid_datagen = ImageDataGenerator(rescale=1. / 255.)


# budowanie generatora z katalogu
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    directory=VALID_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)



# wczytanie wybranego modelu (architektura modelu; który model wybieramy z naszego modułu "models.py")
MODEL_NAME = 'custom_VGG16'
architecture = models.custom_VGG16(input_shape=INPUT_SHAPE, num_classes=2, final_activation='softmax')

# wywołanie funkcji budującej strukture modelu
model = architecture.build()


# kompilacja modelu
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# podsumowanie modelu
model.summary()


# zmienne do treningu
dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
filepath = os.path.join('output', 'model_' + f'{MODEL_NAME}_'+ dt + '.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=8)


# przypisanie wag do klas
tg_class = train_generator.class_indices
print(f'\nClasses: {tg_class}') # {'damaged_cars': 0, 'ok_cars': 1}


train_class_weight = {}
for dir in os.listdir(TRAIN_DIR):
    train_class_weight[dir] = len(os.listdir(os.path.join(TRAIN_DIR, dir))) # {'damaged_cars': 619, 'ok_cars': 973}

damaged_ratio = np.round(train_class_weight['ok_cars'] / train_class_weight['damaged_cars'], 2)
print(damaged_ratio)

class_weight = {tg_class['damaged_cars']: damaged_ratio, tg_class['ok_cars']: 1}
print(class_weight)


print('\nTrenowanie modelu.')
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    verbose=2,
    callbacks=[checkpoint, early_stopping]
)


# eksport wyników do plików
print('\nEksport wykresu do pliku html.')
filename = os.path.join('output', 'report_' f'{MODEL_NAME}_'+ dt + '.html')
plot_hist(history, filename=filename)


# eksport pliku z mapowaniem klas
print('\nEksport etykiet do pliku.')
with open(r'output\labels.pickle', 'wb') as file:
    file.write(pickle.dumps(train_generator.class_indices))


print('\nKoniec')


'''
Przykład uruchomienia:
wpisując odpowiednią liczbę epok i 'Run'
lub w terminalu:
>>>> python 02_train.py -e 20
'''