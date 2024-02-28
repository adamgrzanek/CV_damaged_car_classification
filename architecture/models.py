from tensorflow.keras.models import Sequential # do budowy modelu warstwa po warstwie
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.applications import VGG16
from tensorflow.keras.models import Model



# bęzdiemy importować tą klasę do naszych skryptów
# (dodajemy plik __init__.py w tym katalogu)

class LeNet5:

    # konstruktor (będziemy musieli podać parametr (input_shape) przy tworzeniu instancji modelu)
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        model = Sequential()

        model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D())

        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D())

        # wypłaszczenie
        model.add(Flatten())

        # warstwy gęste
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid')) # units=1: problem binarny;
        # activation='sigmoid': zwróci prawdopodobieństwo (0-1)

        return model



class VGGNetSmall():
    # konstruktor; podejemy 3 parametry
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation # funkcja aktywacji w ostatniej warstwie

    def build(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=1024, activation='relu')) # warstwa gęstopołączona
        model.add(Dropout(0.5))
        model.add(Dense(units=self.num_classes, activation=self.final_activation))

        return model



# pretrained VGG16
class VGG16():

    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation


    def build(self):

        vgg_model = VGG16(weights='imagenet',  # wagi przetrenowana na zbiorze imagenet
                          include_top=False,
                          # czy dołączyć górną cześć sieci, jeżeli True dołączy gęsto połączoną warstwę z 1000 klas
                          input_shape=self.input_shape)  # default 224x224.

        # zamrożenie warstw (nie będą się trenować)
        for layer in vgg_model.layers:
            layer.trainable = False

        # top
        x = vgg_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.1)(x)  # Dropout layer to reduce overfitting
        x = Dense(256, activation='relu')(x)
        x = Dense(self.num_classes, activation=self.final_activation)(x)
        transfer_model = Model(inputs=vgg_model.input, outputs=x)

        return transfer_model


# architecture original VGG16
class VGG16_arch():
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation

    def build(self):
        model = Sequential()

        # default input shape = (224, 224, 3)
        model.add(Conv2D(input_shape=self.input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='vgg16'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(256, activation='relu', name='fc1'))
        model.add(Dense(128, activation='relu', name='fc2'))

        #model.add(Dense(1, activation='sigmoid', name='output'))
        model.add(Dense(units=self.num_classes, activation=self.final_activation))

        return model


class model_v1():

    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation  # funkcja aktywacji w ostatniej warstwie


    def build(self):
        model = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            # 32 filtry/konwolucje o rozmiarze 3x3
            MaxPooling2D(pool_size=(2, 2)),  # rozmiar okienka
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            # przejście na zwykłą sieć MLP
            Flatten(),  # wypłaszczenie
            Dense(units=512, activation='relu'),
            Dense(units=256, activation='relu'),
            Dropout(0.25),
            Dense(units=self.num_classes, activation=self.final_activation)
            # tu units=1 -> problem klasyfikacji binarnej    funkcja sigmoid zwróci prawdopodobieństwo klasy 0 lub 1
            ])

        return model




class ag_v2():

    # konstruktor; podejemy 3 parametry
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation  # funkcja aktywacji w ostatniej warstwie


    def build(self):
        model = Sequential([
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            # 32 filtry/konwolucje o rozmiarze 3x3
            MaxPooling2D(pool_size=(2, 2)),  # rozmiar okienka
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            # przejście na zwykłą sieć MLP
            Flatten(),  # wypłaszczenie
            Dense(units=512, activation='relu'),
            Dense(units=256, activation='relu'),
            Dense(units=128, activation='relu'),
            Dropout(0.2),
            Dense(units=self.num_classes, activation=self.final_activation)
            # tu units=1 -> problem klasyfikacji binarnej    funkcja sigmoid zwróci prawdopodobieństwo klasy 0 lub 1
            ])

        return model
