import os
from imutils import paths # do wylistownia wszystkich obrazów w danym katalogu ****


# cd
base_path = 'cars_image'

print(base_path)
print(os.listdir(base_path))
'''
print(os.getcwd())

print()
for dir in os.listdir(base_path):
    print(dir)
    path = os.path.join(base_path, dir)
    images = paths.list_images(path)
    print(path)
    # print(next(files))
    for n, image in enumerate(images, start=1):
        print(n, image)
        if image.split('.')[-1] in ['jpeg', 'jpg', 'png']:
            #new_image_name = f'{n:04}.jpg'
            new_file_name = os.path.join(path, f'{n:04}.jpg')
            #print(new_file_name)
            os.rename(image, new_file_name)



'''