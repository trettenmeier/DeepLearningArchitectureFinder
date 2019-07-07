# Diese klasse erstellt aus einer Liste mit einem genCode ein Model und
# stellt die Methoden für das Training und Logging der neuronalen Netze
# zur Verfügung

import numpy as np
from time import time
import keras
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import ReLU
from keras.layers import add
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import regularizers


class Modelbuilder:
    def __init__(self, individuum):
        self.individuum = individuum

    # Diese Methode übersetzt den genCode in Layer und erstellt das Model   
    def add_GenCode_To_Model(self):
        ###### (1) statisch ######
        input_img = Input(shape = (56, 56, 3))
        genCodeIterator = 0
        towerCounter = 0
        
        # he initialization der ReLUs
        he_normal = keras.initializers.he_normal(seed=None)

        # Input Layer von VGGNet
        x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(input_img)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
      

       ###### (2) Gene abfragen  ######
        for i in range(0,11):                                       
            # an definierten Stellen Pooling Layer hinzufügen
            if i == 0 or i == 3 or i == 6:
                x = BatchNormalization()(x)
                x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
                continue

            # an allen anderen Stellen den Wert des Gens prüfen und entsprechendes Layer hinzufügen
            gen = self.individuum.genCode[genCodeIterator]
            
            # Inception-Modul (Komplexe Variante)
            if gen == 1 or gen == 16 or gen == 17 or gen == 18 or gen == 19:
                tower = [
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None]

                x = BatchNormalization()(x)
                tower[towerCounter] = Conv2D(64, (1,1), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                tower[towerCounter] = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(tower[towerCounter])
                towerCounter += 1
                tower[towerCounter] = Conv2D(64, (1,1), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                tower[towerCounter] = Conv2D(64, (5,5), padding='same', activation='relu', kernel_initializer=he_normal)(tower[towerCounter])
                towerCounter += 1
                tower[towerCounter] = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
                tower[towerCounter] = Conv2D(64, (1,1), padding='same', activation='relu', kernel_initializer=he_normal)(tower[towerCounter])
                towerCounter += 1

                x = concatenate([tower[towerCounter-3], tower[towerCounter-2], tower[towerCounter-1]], axis = 3)
 
            # ein Conv Layer
            if gen == 2:
                x = BatchNormalization()(x)
                x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                
            if gen == 3:
                x = BatchNormalization()(x)
                x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)

            if gen == 4:
                x = BatchNormalization()(x)
                x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)

            if gen == 5:
                x = BatchNormalization()(x)
                x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)

            if gen == 6:
                x = BatchNormalization()(x)
                x = Conv2D(64, (2,2), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                
            if gen == 7:
                x = BatchNormalization()(x)
                x = Conv2D(128, (2,2), padding='same', activation='relu', kernel_initializer=he_normal)(x)

            if gen == 7:
                x = BatchNormalization()(x)
                x = Conv2D(256, (2,2), padding='same', activation='relu', kernel_initializer=he_normal)(x)

            if gen == 9:
                x = BatchNormalization()(x)
                x = Conv2D(512, (2,2), padding='same', activation='relu', kernel_initializer=he_normal)(x)
            

        	# zwei Conv Layer
            if gen == 10:
                x = BatchNormalization()(x)
                x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                x = BatchNormalization()(x)
                x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                
            if gen == 11:
                x = BatchNormalization()(x)
                x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                x = BatchNormalization()(x)
                x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)

            if gen == 12:
                x = BatchNormalization()(x)
                x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                x = BatchNormalization()(x)
                x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer=he_normal)(x)
               

            # resnet-module 128 (einfach, doppelt, 3fach)
            def resNet128(x):
                x = Conv2D(128, (1,1), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                shortcut = x
                x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = add([shortcut, x])
                x = ReLU()(x)
                return x

            if gen == 13:
                x = resNet128(x)

            if gen == 14:
                x = resNet128(x)
                x = resNet128(x)

            if gen == 15:
                x = resNet128(x)
                x = resNet128(x)
                x = resNet128(x)

            # resnet-module 64 (einfach, doppelt, 3fach)
            def resNet64(x):
                x = Conv2D(64, (1,1), padding='same', activation='relu', kernel_initializer=he_normal)(x)
                shortcut = x
                x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = add([shortcut, x])
                x = ReLU()(x)
                return x

            if gen == 20:
                x = resNet64(x)

            if gen == 21:
                x = resNet64(x)
                x = resNet64(x)

            if gen == 22:
                x = resNet64(x)
                x = resNet64(x)
                x = resNet64(x)
                

            # dann genCodeIterator hochzählen um das nächste Gen zu erreichen
            genCodeIterator += 1

        ###### (3) statische Layer am Ende ######
        x = BatchNormalization()(x)
        x = Flatten()(x)

        x = Dense(1024, activation='relu', kernel_initializer=he_normal)(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(1024, activation='relu', kernel_initializer=he_normal)(x)
        x = Dropout(rate=0.2)(x)

        out = Dense(200, activation='softmax')(x)
         
        self.model = Model(inputs = input_img, outputs = out)

    # Methode um das Model zu kompilieren + Optimizer + Visualisierung
    ###### Wenn Fehler auftreten: Die Try-Except-Blöcke entfernen, dann gibt Python die Fehlermeldung aus ######
    def finalize_Model(self, epochen):
        # Optimierer definieren
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        try:
            # Compile
            self.model.compile(
                loss=keras.losses.categorical_crossentropy, 
                optimizer=adam,
                metrics=['accuracy']
                )
            
            # Model visualisieren
            plot_model(self.model, to_file='output/zz_model_plot_latest.png', show_shapes=True, show_layer_names=True)
            self.model.summary()

            return self.train_Model(epochen)

        except (KeyboardInterrupt, SystemExit): 
            raise
        except:
            return -1   # geringster Fitnesswert im Fehlerfall um die Selektion des Individuums zu vermeiden
        finally:
            K.clear_session()   # auf jeden fall die GPU-Ressourcen wieder freigeben!
       
    # Methode für das Training
    def train_Model(self, epochen):
        batch_size = 56
        
        # Plateaubasierte Reduzierung der Lernrate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=5, 
            verbose=1, 
            cooldown=0, 
            min_lr=0)

        # Visualisierung hinzufügen
        tensorboard = TensorBoard(log_dir="output/logs/{}".format(time()))

        # Data Augmentation (datagen=>Trainingsdaten, datagen2=>Validierungsdaten)
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,                               
            rotation_range=40,
            #width_shift_range=0.2,       
            #height_shift_range=0.2,    
            #rescale=1./255,              
            #shear_range=0.2,           
            #zoom_range=0.2,             
            horizontal_flip=True,
            fill_mode='nearest'
        )

        datagen2 = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True
        )

        generator = datagen.flow_from_directory(
            'tiny-imagenet-200/train',
            target_size=(64,64),
            batch_size=batch_size,
            class_mode="categorical"
        )

        generator2 = datagen2.flow_from_directory(
            'tiny-imagenet-200/val',
            target_size=(64,64),
            batch_size=batch_size,
            class_mode="categorical"
        )

        # Zuschneiden der Bilder
        def random_crop(img, random_crop_size):
            assert img.shape[2] == 3
            height, width = img.shape[0], img.shape[1]
            dy, dx = random_crop_size
            x = np.random.randint(0, width - dx + 1)
            y = np.random.randint(0, height - dy + 1)
            return img[y:(y+dy), x:(x+dx), :]


        def crop_generator(batches, crop_length):

            while True:
                batch_x, batch_y = next(batches)
                batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
                for i in range(batch_x.shape[0]):
                    batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
                yield (batch_crops, batch_y)

        train_crops = crop_generator(generator, 56)
        valid_crops = crop_generator(generator2, 56)

        # Mit Jupyter Notebook berechnete Mittelwerte + Standardabweichung der RGB-Kanäle (nur Trainingsdaten!)
        datagen.mean = np.array([122.4626756 , 114.25840613, 101.37467571], dtype=np.float32).reshape((1,1,3))
        datagen.std = 70.93752549462285

        # Die Werte der Trainingsdaten müssen auch für die Validierungsdaten übernommen werden. 
        datagen2.mean = np.array([122.4626756 , 114.25840613, 101.37467571], dtype=np.float32).reshape((1,1,3))
        datagen2.std = 70.93752549462285

        # Training
        history = History()

        self.model.fit_generator(
            train_crops,
            verbose=1,
            epochs=epochen,
            steps_per_epoch=100000/batch_size,
            validation_data=valid_crops,
            validation_steps=10000/batch_size,
            callbacks=[tensorboard, history, reduce_lr]
        )

        # Log sichern, zusätzlich zum Tensorboard
        historyfile = open("output/zz_log_history.txt", "a+")
        historyfile.write("\n")
        historyfile.write("Logeintrag zur ID-Nr: " + str(self.individuum.id) + " (loss, acc, val_loss, val_acc)")
        historyfile.write("\n")
        historyfile.write(str(history.history['loss']) + "\n")
        historyfile.write(str(history.history['acc']) + "\n")       
        historyfile.write(str(history.history['val_loss']) + "\n")        
        historyfile.write(str(history.history['val_acc']) + "\n")
        historyfile.write("\n")
        historyfile.close()

        # Model speichern
        self.model.save('output/zz_model_latest.h5')

        return self.evaluate_Model(valid_crops, batch_size)

    # Fitness berechnen und zurückgeben
    def evaluate_Model(self, valid_crops, batch_size):
        score = self.model.evaluate_generator(
            valid_crops, 
            steps=10000/batch_size, 
            verbose=1)
        return score[1]
