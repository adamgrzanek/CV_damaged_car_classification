import os
from imutils import paths


base_path = 'cars_image'

print(base_path)
print(os.listdir(base_path))
print(os.getcwd())

print('\nZmiana nawz plików w katalogach będących w bazowym folderze')

for dir in os.listdir(base_path):
    print(dir)
    path = os.path.join(base_path, dir)
    images = paths.list_images(path)
    for n, image in enumerate(images, start=1):
        if image.split('.')[-1] in ['jpeg', 'jpg', 'png']:
            new_file_name = os.path.join(path, f'{n:04}.jpg')
            os.rename(image, new_file_name)
