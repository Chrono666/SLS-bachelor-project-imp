import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob


def load_model(model_path, weight_path):
    # load json and create model
    file = open(model_path, 'r')
    model_json = file.read()
    file.close()
    loaded_model = keras.models.model_from_json(model_json)
    # load weights
    loaded_model.load_weights(weight_path)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    loaded_model.compile(loss="binary_crossentropy", optimizer=optimizer,
                         metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
    return loaded_model


def init_model(json_path, weights_path):
    return load_model(json_path, weights_path)


def classify_single_image(image_path, model):
    original = cv2.imread(image_path)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (224, 224)) / 255
    image = np.array([img])
    vgg_input = keras.applications.vgg16.preprocess_input(image * 255)
    Y_prob = model.predict(vgg_input)
    if Y_prob > 0.8:
        print('The image belongs to the ok class', Y_prob)
    else:
        print('The image belongs to the def class', Y_prob)


# "data/data_predictions/OK/*.jpg"
def load_images_from_folder(file_path):
    x_data = []
    files = glob.glob(file_path)
    for myFile in files:
        i = cv2.imread(myFile)
        x_data.append(i)
    return x_data


def classify_multiple_images(folder_path, model):
    images = load_images_from_folder(folder_path)
    images_array = np.array(images)
    images_resized = tf.image.resize(images_array, [224, 224])
    model_input = keras.applications.vgg16.preprocess_input(images_resized * 255)
    Y_proba = model.predict(model_input)
    for prediction in Y_proba:
        if prediction > 0.8:
            print('image is ok', prediction)
        else:
            print('image is defect', prediction)


main_menu_options = {
    1: 'classify single image',
    2: 'classify folder with images',
    3: 'exit',
}


def print_menu():
    for key in main_menu_options.keys():
        print(key, '--', main_menu_options[key])


def option1(image_path, model):
    classify_single_image(image_path, model)


def option2(folder_path, model):
    classify_multiple_images(folder_path, model)


if __name__ == '__main__':

    # json_path = input('Please enter json path for initializing the model: \n')
    # weight_path = input('Please enter h5 path for initializing the model: \n')
    # model = init_model(json_path, weight_path)
    model = init_model('saved_models/no_augmentation/full_data/second_train_step/vgg16.json',
                       'saved_models/no_augmentation/full_data/second_train_step/vgg16_weights.h5')

    while (True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        # Check what choice was entered and act accordingly
        if option == 1:
            image_path = input("Please enter the path of the image: \n")
            option1(image_path, model)
        elif option == 2:
            folder_path = input("Please enter the folder path of the images: \n")
            option2(folder_path, model)
        elif option == 3:
            print('Script will close')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 3.')
