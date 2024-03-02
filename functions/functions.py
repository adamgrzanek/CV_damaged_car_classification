import numpy as np
import tensorflow as tf
import cv2
import imutils
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import cm
import matplotlib.pyplot as plt
import torch

# heatmap
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam


# pobranie biblioteki i plików do segmentacji
# wpisać w terminalu:
# pip install 'git+https://github.com/facebookresearch/segment-anything.git'

# pobierz plik z poniższego adresu:
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Następnie przenieś go do katalogu "CV_damaged_car_classification/functions/SAM_segmentation"
# w wyniku powinniśmy otrzymać plik "sam_vit_h_4b8939.pth" w tym katalogu


# pobranie pliku do detekcji objektów
# pobierz plik z poniższego adresu:
# https://pjreddie.com/media/files/yolov3.weights
# Następnie przenieś go do katalogu "CV_damaged_car_classification/functions/YOLO_detection"
# w wyniku powinniśmy otrzymać plik "yolov3.weights" w tym katalogu


#############################################################
#sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "functions/SAM_segmentation/sam_vit_h_4b8939.pth"
model_type = "vit_h"

if torch.cuda.is_available() == True:
    device = "cuda"
else:
    device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
##############################################################



# funkcje SAM
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))




def segment_car(img, box, margin=0):
    '''
    Funkcja do segmentacji objektu w ramce.
    Parametrem jest zdjęcie i ramka. Opcjonalnie można dodać margines do powiększenia ramki
    Funkcja zwraca maske o wymiarach zdjęcia i wartościach True i False.
    '''
    try:
        image = cv2.imread(img)
    except:
        image = img
    #cv2_imshow(image)
    predictor.set_image(image)

    if margin:
        box = [box[0]-margin, box[1]-margin, box[2]+margin, box[3]+margin]

    input_box = np.array(box) # box from get_car_image function

    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.title('Wykryty objekt')
    plt.show()

    return masks[0]



def find_min_max(x):
    '''
    Funkcja do zwracania pozycji pierwszego i ostatniego niezerowego elementu w objekcie iterowalnym.
    (Do znalezienie konturów na masce obrazu)
    przykład: X = [0, 0, 4, 8, 2, 6, 0, 0]
    find_min_max(X) -> 2, 6
    '''
    for i in range(len(x)):
        if x[i] > 0:
            min = i
            break
    for j in range(len(x)):
        if x[-j] != 0:
            max = len(x) - j
            break
    return min, max




def clean_image(img, mask):
    '''
    Funkcja do nałożenia maski (o wartościach True/False) na obraz oraz wykadrowania danego objektu.
    Wynikiem jest wyfiltrowany i wykadrowany objekt na białym tle
    '''
    try:
        new_image = cv2.imread(img)
    except:
        new_image = img

    # nałożenie maski
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            if mask[h][w] == False:
                new_image[h][w] = [255, 255, 255] # białe tło

    # szukanie skrajnych punktów (na masce o wartościach 0 i 1)
    mask_int = mask.astype(int)
    vertical_search = np.sum(mask_int, axis=0) # szuka w poziomie (punkt lewy/prawy)
    horizontal_search = np.sum(mask_int, axis=1) # szuka w pionie (punkt górny/dolny)

    left, right = find_min_max(vertical_search)
    top, bottom = find_min_max(horizontal_search)

    cutted_image = new_image[top:bottom, left:right]
    cv2.imshow('Wynikowy obraz', cutted_image)
    cv2.waitKey(0)
    return cutted_image



def get_car_image(image_path,
                  weights_path= 'functions/YOLO_detection/yolov3.weights',
                  config_path = 'functions/YOLO_detection/yolov3.cfg',
                  labels = 'functions/YOLO_detection/coco.names',
                  CONFIDENCE = 0.6, THRESHOLD = 0.3, margin=20, show_data=True):
    '''
    Funkcja do detekcji objektów (pojazdów).
    Argumentem jest ścieżka do zdjęcia.
    Na wyjściu otrzymamy obraz z wykrytymi pojazdami oraz wykadrowany główny pojazd (z ramką do segmantacji)
    '''
    LABELS = open(labels).read().strip().split('\n')
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    image = cv2.imread(image_path)

    (h, w) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1/255., (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    #start = time.time()
    layer_outputs = net.forward(ln)
    #end = time.time()


    boxes = []
    confidences = []
    class_ids = []
    cars = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([w, h, w, h])
                (x_center, y_center, width, height) = box.astype('int')

                x = int(x_center - (width / 2))
                y = int(y_center - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)


    image_copy = image.copy()
    image_with_boxes = image.copy()
    if len(idxs) > 0:
        for n, i in enumerate(idxs.flatten(), start=1):

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_ids[i]]]

            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2) # original box
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2) # original box

            text = f'{LABELS[class_ids[i]]}: {confidences[i]:.4f}    number: {n}'
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(image_with_boxes, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if LABELS[class_ids[i]] == 'car' or LABELS[class_ids[i]] == 'truck' :
                area = w * h
                #if area >= 100000: # aby uniknąć małych aut w tle
                cars.append({'confidence': confidences[i], 'area': area, 'x': x, 'y': y, 'w': w, 'h': h})
                #image_with_boxes = imutils.resize(image_with_boxes, width=600)
        cv2.imshow('Wykryte objekty', image_with_boxes)


    if cars: # jeśli znalanł pojazdy
        main_car = sorted(cars, key=lambda x: x['area'], reverse=True)[0] # kolejność po największym obszarze
        x, y, w, h = main_car['x'], main_car['y'], main_car['w'], main_car['h'] # original box
        # if show_data:
        #     print(cars)

        # expanded box
        margin = margin
        y_start = y - margin if y - margin >= 0 else 0 # np.clip
        y_end = y + h + margin if y + h + margin <= image.shape[0] else image.shape[0]
        x_start = x - margin if x - margin >= 0 else 0
        x_end = x + w + margin if x + w + margin <= image.shape[1] else image.shape[1]

        expanded_box = [x_start, y_start, x_end, y_end]
        main_car_image = image_copy[y_start : y_end, x_start : x_end]

    image = imutils.resize(image, width=600) # zmiana rozmiaru z zachowaniem proporcji
    if show_data:
        cv2.imshow('Bazowy obraz ', image) # oryginalne zdjęcie po resize

    if cars:
        # print('====')
        main_car_image_resized = imutils.resize(main_car_image, width=600)
        if show_data:
            #print('-' * 80)
            cv2.imshow('Główny pojazd', main_car_image_resized)
            cv2.waitKey(0)
        return main_car_image, expanded_box



def prepare_image(img, input_shape=(224, 224, 3)):
    '''
    Funkcja do przygotowania zdjęcia do klasyfikacji
    '''
    try:
        image = cv2.imread(img)
    except:
        image = img
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = image.astype('float') / 255.
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


def show_heatmap(img, model, labels_path):

    # przygotowane zdjęcia
    image = prepare_image(img)

    # predykcja
    y_pred = model.predict(image)[0]

    # wczytanie etykiet
    with open(labels_path, 'rb') as file:
        classes = pickle.loads(file.read()) # {'damaged_cars': 0, 'ok_cars': 1}

    labels = dict(enumerate(classes)) # {0: 'damaged_cars', 1: 'ok_cars'}


    pred_class = np.argmax(y_pred)
    text = f'{labels[pred_class]}, prob: {100 * y_pred[pred_class]:.2f}%'



    # {'damaged_cars': 0, 'ok_cars': 1}
    score = CategoricalScore(pred_class)
    input_class = 'ok_car' if pred_class == 1 else 'damaged_car'


    gradcam = Gradcam(model,
                      model_modifier=model_modifier, # zmienia funkcję aktywacji w ostatniej warstwie modelu
                      clone=False)

    # Generowanie heatmapy z GradCAM
    cam = gradcam(score,
                image,
                penultimate_layer=-1)


    # obraz
    fig, ax = plt.subplots()
    heatmap = np.uint8(cm.jet(cam[0])[..., :4] * 255)
    ax.set_title(text, fontsize=16)
    ax.imshow(image[0])
    ax.imshow(heatmap, cmap='jet', alpha=0.5) # nakładka
    ax.axis('off')
    plt.show()



def model_modifier(mdl):
    '''
    Funkcja do zmiany funkcji atywacji (na liniową) w ostatniej warstwie modelu
    '''
    mdl.layers[-1].activation = tf.keras.activations.linear