from functions.functions import get_car_image, segment_car, clean_image,show_heatmap
from tensorflow.keras.models import load_model



img_path = 'example_images\damaged_01.jpg'
model_path = 'output\model_custom_VGG16_2_classes.hdf5'
labels_path = 'output\labels_2_classes.pickle'

main_car_image, expanded_box = get_car_image(img_path, show_data=True)
box = expanded_box
mask = segment_car(img_path, box, margin=0)
clean_car = clean_image(img_path, mask)


loaded_model = load_model(model_path)
loaded_model.summary()


show_heatmap(clean_car, loaded_model, labels_path)