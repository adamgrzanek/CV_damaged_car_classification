import argparse
from functions.functions import get_car_image, segment_car, clean_image,show_heatmap
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# parametry przy uruchamianiu
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
ap.add_argument('-mar', '--margin', required=False, help='image margin for segmentation', default=0)
ap.add_argument('-m', '--model', required=False, help='path to model', default='output\model_custom_VGG16_2_classes.hdf5')
ap.add_argument('-l', '--labels', required=False, help='path to labels', default='output\labels_2_classes.pickle')
args = vars(ap.parse_args())
print(args)

img_path = args['image']
model_path = args['model']
labels_path = args['labels']
margin = int(args['margin'])

print('[INFO] Detekcja obiektów i wyznaczenie ramki na główym pojeździe')
main_car_image, expanded_box = get_car_image(img_path, show_data=True)

print('[INFO] Segmentacja wybranego obiektu i wydobycie maski')
mask = segment_car(img_path, expanded_box, margin=margin)

print('[INFO] Czyszczenie obrazu')
clean_car = clean_image(img_path, mask)

print('[INFO] Załadowanie modelu')
loaded_model = load_model(model_path)

print('[INFO] Predykcja i mapa ciepła')
show_heatmap(clean_car, loaded_model, labels_path)

print('[INFO] Koniec')


'''
Przykład uruchomienia w terminalu:
>>>> python 07_predict_with_heatmap.py -i example_images/damaged_01.jpg
'''