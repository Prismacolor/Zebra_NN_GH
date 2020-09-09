import os
import pandas
import numpy
import PIL
from PIL import ImageFile

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# project source: http://lila.science/

# part one, gathering and processing the data #
base_image_list = []

working_id_list = []
matched_img_list = []
second_id_list = []
species_list = []

final_id, final_img, final_species = [], [], []

data_folder = r'C:\Zebra_NN\KRU_S1'

# place images from data folder into a master list
for root, d_names, f_names in os.walk(data_folder):
    for name in f_names:
        if name.endswith('.JPG'):
            path_image = os.path.join(root, name)
            path_img_adjusted = path_image.replace('\\', '/')
            base_image_list.append(path_img_adjusted)

# upload source CSVs, and read them into pandas for conversion
image_paths = r'../KRU_S1_report_lila_image_inventory.csv'
report_path = r'../KRU_S1_report_lila.csv'

image_path_read = pandas.read_csv(image_paths)
report_path_read = pandas.read_csv(report_path)

# both dataframes have a capture id column, but one has the image ids and the other has the categories
# we need to merge those two together, so we first remove duplicate ids from both data sets
# this allows us to join them together so the ids, images, and categories match up
working_df_image = pandas.DataFrame(image_path_read, columns=['capture_id', 'image_path_rel'])
working_df_species = pandas.DataFrame(report_path_read,  columns=['capture_id', 'question__species'])


# part two: initial processing and merging of data #
# match images in dataset with images created in setup
def match_images(base_list, img_df):
    for item in img_df.itertuples():
        for image in base_list:
            if image.endswith(item[2]):
                working_id_list.append(item[1])
                matched_img_list.append(item[2])

    return working_id_list, matched_img_list


def get_species(id_list, species_df):
    for item in species_df.itertuples():
        if item[1] in id_list:
            second_id_list.append(item[1])
            species_list.append(item[2])

    return second_id_list, species_list


def merge_lists(ids, sec_ids, images, species):
    for item1, item2 in zip(id_list, img_list):
        for item3, item4 in zip(sec_id_list, species_match_list):
            if item1 == item3:
                final_id.append(item3)
                final_img.append(item2)
                final_species.append(item4)

    # print(len(final_id), len(final_img), len(final_species))

    return final_id, final_img, final_species


def create_encoders(dataframe):
    y0 = dataframe['question__species'].to_numpy().reshape(-1, 1)
    one_hot = preprocessing.OneHotEncoder()

    y = one_hot.fit_transform(y0)

    categories = one_hot.categories_
    print(categories)

    return dataframe['image_path_rel'], y


id_list, img_list = match_images(base_image_list, working_df_image)
sec_id_list, species_match_list = get_species(id_list, working_df_species)
final_id_list, final_img_list, final_species_list = merge_lists(id_list, sec_id_list, img_list, species_match_list)

final_data_dict = {'capture_id': final_id_list, 'image_path_rel': final_img_list, 'question__species': final_species_list}
final_working_df = pandas.DataFrame.from_dict(final_data_dict, orient='columns')

# now create our X and y, and split the dataset.
X, y = create_encoders(final_working_df)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=39)

# part III, let's make a neural network! #
# we need to do some additional processing of the images
num_features = 2352


def image_as_array(dataset):
    # kept receiving file truncated errors, this appears to be an issue with PIL, this line should fix it
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image_arrays = []
    if isinstance(dataset, pandas.Series):
        for item in dataset:
            try:
                converted_item = r'../' + item
                image = tf.keras.preprocessing.image.load_img(
                    converted_item, grayscale=False, color_mode="rgb", target_size=(28, 28), interpolation="nearest")
                input_arr = keras.preprocessing.image.img_to_array(image)
                input_arr = numpy.array([input_arr])
                image_arrays.append(input_arr)
            except Exception as e:
                print(item, e)
                continue

        dataset['Image_Array'] = image_arrays
        return dataset
    elif isinstance(dataset, list):
        for item in dataset:
            try:
                image = tf.keras.preprocessing.image.load_img(item)
                input_arr = keras.preprocessing.image.img_to_array(image)
                input_arr = numpy.array([input_arr])
                image_arrays.append(input_arr)
            except Exception as e:
                print(item, e)
                continue

        return image_arrays


# convert and reshape everything to the right format for keras
X_train, x_test = image_as_array(X_train), image_as_array(x_test)
X_train, x_test = X_train['Image_Array'], x_test['Image_Array']

X_train, x_test = numpy.array(X_train, numpy.float32), numpy.array(x_test, numpy.float32)
y_train, y_test = y_train.toarray(), y_test.toarray()
X_train, x_test = X_train.reshape((-1, num_features)), x_test.reshape((-1, num_features))

# building the model with multiple drop out layers to avoid overfitting
img_model = Sequential()
img_model.add(Dense(512, input_dim=num_features, activation='relu'))
img_model.add(Dropout(.25))
img_model.add(Dense(225, activation='relu'))
img_model.add(Dropout(.25))
img_model.add(Dense(31, activation='softmax'))

opt = SGD(lr=0.01)

img_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

print('fit model')
img_model.fit(X_train, y_train, epochs=10, verbose=2, validation_split=0.2, validation_data=(x_test, y_test))

score = img_model.evaluate(x_test, y_test, verbose=2)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
