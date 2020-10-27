import numpy as np
import sys
from tqdm import *

# Extracting vgg 16 features
# from https://github.com/XifengGuo/DEC-keras

def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet',
                  input_shape=(im_h, im_h, 3))

    feature_model = Model(model.input, model.get_layer('fc1').output)
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,
                                                                       im_h))) for im in x])
    x = preprocess_input(x)
    features = feature_model.predict(x)

    return features

VGG_16_FEATURES = 4096

def extract(x):
    # extract features
    features = np.zeros((len(x), VGG_16_FEATURES))
    BATCHES = 6
    SLICE = len(x) / BATCHES
    for i in tqdm(range(BATCHES)):
        idx = range(i * SLICE, (i + 1) * SLICE)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)
    return features


X = np.load(sys.argv[1])
X = X['train_data']

X = np.array(X * 255).astype(int)
X = X.reshape(len(X), 64, 64, 3)

_X = extract(X)

np.savez_compressed('GHTVGG16.npz', train_data=_X)
