import glob
import os
import random

import numpy as np
import pandas as pd
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


def read_img(imgpath, size):
    img = image.load_img(imgpath, target_size=size)
    img = image.img_to_array(img)
    return img


def extract_features(model, paths):
    images = np.zeros((len(paths), 299, 299, 3), dtype='float32')
    for i, imgpath in enumerate(paths):
        img = read_img(imgpath, (299, 299))
        x = preprocess_input(np.expand_dims(img.copy(), axis=0))
        images[i] = x
    feats = model.predict(images, batch_size=32, verbose=1)
    return feats


if __name__ == "__main__":
    train_dir = 'data/train'
    valid_dir = 'data/valid'
    test_dir = 'data/test'

    # load file paths
    train_paths = glob.glob(os.path.join(train_dir, '*', '*.jpg'))
    valid_paths = glob.glob(os.path.join(valid_dir, '*', '*.jpg'))
    test_paths = glob.glob(os.path.join(test_dir, 'unknown', '*.jpg'))
    random.shuffle(train_paths)

    # load pretrained model
    xception_bottleneck = Xception(weights='imagenet', include_top=False, pooling='avg')
    xception_bottleneck.summary()

    # extract Xception bottleneck features
    if not os.path.exists('train_feats_xception.npy'):
        train_feats = extract_features(xception_bottleneck, train_paths)
        valid_feats = extract_features(xception_bottleneck, valid_paths)
        test_feats = extract_features(xception_bottleneck, test_paths)

        # save features
        np.save('train_feats_xception.npy', train_feats)
        np.save('valid_feats_xception.npy', valid_feats)
        np.save('test_feats_xception.npy', test_feats)
    else:
        train_feats = np.load('train_feats_xception.npy')
        valid_feats = np.load('valid_feats_xception.npy')
        test_feats = np.load('test_feats_xception.npy')

    # normalize features
    scaler = StandardScaler().fit(train_feats)
    train_feats_norm = scaler.transform(train_feats)
    valid_feats_norm = scaler.transform(valid_feats)
    test_feats_norm = scaler.transform(test_feats)

    # make labels
    dogs = list(set([x.split('/')[2] for x in train_paths]))
    dogs = sorted(dogs)

    train_labels = [dogs.index(x) for x in [x.split('/')[2] for x in train_paths]]
    valid_labels = [dogs.index(x) for x in [x.split('/')[2] for x in valid_paths]]

    # for calculating validation accuracy
    valid_onehot = np_utils.to_categorical(valid_labels)

    # train and validation
    logreg = LogisticRegression(multi_class='multinomial', random_state=1234)
    logreg.fit(train_feats_norm, train_labels)

    valid_probs = logreg.predict_proba(valid_feats_norm)
    valid_preds = logreg.predict(valid_feats_norm)

    print('Validation LogLoss {}'.format(log_loss(valid_onehot, valid_probs)))
    print('Validation Accuracy: {}'.format(accuracy_score(valid_labels, valid_preds)))

    # test
    test_probs = logreg.predict_proba(test_feats_norm)  # (10357, 120)
    test_probs = test_probs.T  # (120, 10357)

    # 列ごとにDataFrameへデータを格納
    data = {}
    data['id'] = [x.split('/')[3].replace('.jpg', '') for x in test_paths]
    for i, dog in enumerate(dogs):
        data[dog] = test_probs[i]
    submissions = pd.DataFrame(data=data, columns=['id'] + dogs)

    os.makedirs('submits', exist_ok=True)
    submissions.to_csv('submits/logreg_xception_bottleneck.csv', index=False, header=True)
