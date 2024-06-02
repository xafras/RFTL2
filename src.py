###################################################################################################
#   ARCHITECTURE

"""
RFTL/
|   transfert.ipynb
|   training.ipynb
|   src.py
|   readme.md
|   live.ipynb
|   lfw-names.txt
|   init.ipynb
|   .structure_generator
|   .lfw
`-- .venv/
`-- data/
|   `-- anchor/
|   |   |   2b62794c-1d4a-11ef-b7aa-c0e92b3a9e25.jpg
|   |   |   ...
|   `-- negative/
|   |   |   Aaron_Eckhart_0001.jpg
|   |   |   ...
|   `-- positive/
|   |   |   3a6f5170-1d4a-11ef-b4da-c0e92b3a9e25.jpg
|   |   |   ...
|   `-- var_anchor/
|   |   |   2b62794c-1d4a-11ef-b7aa-c0e92b3a9e25.jpg
|   |   |   ...
|   `-- var_negative/
|   |   |   Aaron_Eckhart_0001.jpg
|   |   |   ...
|   `-- var_positive/
|       |   3a6f5170-1d4a-11ef-b4da-c0e92b3a9e25.jpg
|       |   ...
`-- saved_model/
|   |   model.h5
|   |   model_descriptor.txt
|   |   model_training_overview.xls
`-- lfw/
    |   ...
"""

###################################################################################################
#   PACKAGES

#   Standard packages
import os
import cv2
import uuid
import tqdm
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt

#   Machine learning packages (tensorflow API)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf


###################################################################################################
#   GLOBAL VARIABLES

STR_LOCK = ".structure_generator"
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
VPOS_PATH = os.path.join('data', 'var_positive')
VNEG_PATH = os.path.join('data', 'var_negative')
VANC_PATH = os.path.join('data', 'var_anchor')
VER_PATH = os.path.join('data', 'verification')
BANC_PATH = os.path.join('data', 'benchmark_anchor')
BPOS_PATH = os.path.join('data', 'benchmark_positive')

LFW_LOCK = ".lfw"
LFW_PATH = "lfw"

SVM_PATH = "saved_model"

SAVED_MODEL_PATH = "saved_model"

VERIFICATION_THRESHOLD = 0.55

###################################################################################################
#   UTILITY


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def plot_images_b2b(img1, img2, title1=None, title2=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax1.set_title(title1) if title1 != None else None
    ax2.set_title(title2) if title2 != None else None
    plt.show()


def empty_folder(folder_path):
    if os.path.exists(folder_path):
        for file_path in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, file_path))


def copy_folder(folder_path_src, folder_path_dst):
    assert os.path.exists(folder_path_src) and os.path.exists(folder_path_dst)
    for file_path in os.listdir(folder_path_src):
        EX_PATH = os.path.join(folder_path_src, file_path)
        NEW_PATH = os.path.join(folder_path_dst, file_path)
        shutil.copy(EX_PATH, NEW_PATH)


def delete_folder(folder_path):
    empty_folder(folder_path)
    if os.path.exists(folder_path):
        os.rmdir(folder_path)


###################################################################################################
#   STRUCTURE GENERATION


def generate_structure():
    """ Generate the folder structure of the project. """
    if os.path.exists(STR_LOCK):
        print("[DEBUG] Folder structure already generated")
        return None
    touch(STR_LOCK)
    os.makedirs(POS_PATH, True)
    os.makedirs(NEG_PATH, True)
    os.makedirs(ANC_PATH, True)
    os.makedirs(VPOS_PATH, True)
    os.makedirs(VNEG_PATH, True)
    os.makedirs(VANC_PATH, True)
    os.makedirs(SVM_PATH, True)
    os.makedirs(SAVED_MODEL_PATH, True)
    print("[DEBUG] Folder structure generated")


###################################################################################################
#   DATA COLLECTION & SETUP


def setup_data():
    #   TODO implement automatic database download
    if os.path.exists(LFW_LOCK):
        print("[DEBUG] Data already imported")
        return None
    touch(LFW_LOCK)
    # %wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
    # %tar -xf lfw.tgz
    # %rm -f lfw.tgz
    for directory in tqdm.tqdm(os.listdir(LFW_PATH)):
        for file in os.listdir(os.path.join(LFW_PATH, directory)):
            EX_PATH = os.path.join(LFW_PATH, directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            shutil.copy(EX_PATH, NEW_PATH)
    # %rm -fr lfw
    print("[DEBUG] Finished data setup")


def get_sorted_lfw_table():
    LFW_TEXT_PATH = 'lfw-names.txt'
    LFW_TABLE = dict()
    with open(LFW_TEXT_PATH, 'r') as f:
        for line in f:
            val = line.strip().split('\t')
            LFW_TABLE[val[0]] = int(val[1])
    return sorted(LFW_TABLE.items(),  key=lambda x: x[1], reverse=True)


LFW_NAME_LIST_SORTED = get_sorted_lfw_table()


def setup_var_data(n_th_iteration, max_harvest=100):
    selected_name = LFW_NAME_LIST_SORTED[n_th_iteration][0]
    print(f"Generating data variant with {selected_name} as anchor")
    for directory in tqdm.tqdm(os.listdir(LFW_PATH)):
        if directory == selected_name:
            for file in os.listdir(os.path.join(LFW_PATH, selected_name))[:max_harvest]:
                EX_PATH = os.path.join(LFW_PATH, selected_name, file)
                NEW_PATH = os.path.join(VANC_PATH, file)
                shutil.copy(EX_PATH, NEW_PATH)
        else:
            for file in os.listdir(os.path.join(LFW_PATH, directory))[:max_harvest]:
                EX_PATH = os.path.join(LFW_PATH, directory, file)
                NEW_PATH = os.path.join(VNEG_PATH, file)
                shutil.copy(EX_PATH, NEW_PATH)
    print("[DEBUG] Finished data variant setup")


###################################################################################################
#   VIDEO CAPTURE


def start_video_capture(camera_index=0, awaited_capture=100):
    """ Starts the camera capture. """
    print("Starting video capture\nKeys are:\n- R to refresh\n- A to collect anchor\n- P to collect positive\n- Q to quit.")
    counter = 0
    # establish a connection to the webcam
    cap = cv2.VideoCapture(camera_index)
    while cap.isOpened() and counter < awaited_capture:
        ret, frame = cap.read()
        frame = frame[120:120+250, 200:200+250, :]  # crop frame to 250x250px

        if cv2.waitKey(1) & 0XFF == ord('r'):
            print("R key pressed : Refreshing")

        if cv2.waitKey(1) & 0XFF == ord('a'):
            print("A key pressed : Anchor collected")
            time.sleep(2)
            imgname = os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")
            counter += 1
            cv2.imwrite(imgname, frame)

        cv2.imshow('Image Collection', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            print("Q key pressed : Quitting")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(
        f"Video capture stopped successfully. Took {counter} new anchor.")


###################################################################################################
#   IMAGE PREPROCESSING


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)   # Read in image from file path
    img = tf.io.decode_jpeg(byte_img)       # Load in the image
    # Resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0                       # Scale image to be between 0 and 1
    return img


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

def preprocess_twin_verify(input_img, validation_img):
    return (preprocess(input_img), preprocess(validation_img))


def harvest_from_folder(anc_path=ANC_PATH, pos_path=POS_PATH, neg_path=NEG_PATH, n_harvest=None):
    print(
        f"Harvesting data from anchor {anc_path}, positive {pos_path} and negative {neg_path} folder.")
    if not os.listdir(anc_path):
        raise Exception(f"{anc_path} is empty.")
    if not os.listdir(pos_path):
        raise Exception(f"{pos_path} is empty.")
    if not os.listdir(neg_path):
        raise Exception(f"{neg_path} is empty.")
    anchor = tf.data.Dataset.list_files(anc_path+'\*.jpg')
    positive = tf.data.Dataset.list_files(pos_path+'\*.jpg')
    negative = tf.data.Dataset.list_files(neg_path+'\*.jpg')

    max_anc = len(list(anchor.as_numpy_iterator()))
    max_pos = len(list(positive.as_numpy_iterator()))
    max_neg = len(list(negative.as_numpy_iterator()))
    if n_harvest == None:
        n_harvest = min(max_anc, max_pos, max_neg)
    else:
        n_harvest = min(max_anc, max_pos, max_neg, n_harvest)

    print(f"Anchor images availlable :\t{max_anc}")
    print(f"Positive images availlable :\t{max_pos}")
    print(f"Negative images availlable :\t{max_neg}")
    print(f"Tuple harvested :\t\t{n_harvest}")

    #   Image directories
    anchor = anchor.take(n_harvest)
    positive = positive.take(n_harvest//2)
    negative = negative.take(n_harvest//2 + 1)

    print("Labelling data")
    #   Labelling
    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)
    print(f"Successfully harvested and labelled {n_harvest} data.")
    return data


def preprocess_example(anc_path=ANC_PATH, pos_path=POS_PATH, neg_path=NEG_PATH):
    data = harvest_from_folder(anc_path, pos_path, neg_path)
    samples = data.as_numpy_iterator()
    example = samples.next()
    # for _ in range(len(os.listdir(POS_PATH))):
    #     example = samples.next()
    res = preprocess_twin(*example)
    plot_images_b2b(res[0], res[1])


def data_augment_generate_from_image(img, data_augmentation_factor=2):
    new_data = []
    for i in range(data_augmentation_factor):
        img = tf.image.stateless_random_brightness(
            img, max_delta=0.02, seed=(1, 2))
        img = tf.image.stateless_random_contrast(
            img, lower=0.6, upper=1, seed=(1, 3))
        img = tf.image.stateless_random_flip_left_right(
            img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(
            np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(
            img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
        new_data.append(img)
    return new_data


def data_augment_from_folder(folder_path_in, folder_path_out, data_augmentation_factor=2):
    print(
        f"Preparing data augmentation from {folder_path_in} to {folder_path_out}.")
    counter = 0
    for file_name in tqdm.tqdm(os.listdir(os.path.join(folder_path_in))):
        img_path = os.path.join(folder_path_in, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_augment_generate_from_image(
            img, data_augmentation_factor)
        for image in augmented_images:
            cv2.imwrite(os.path.join(folder_path_out,
                        f"{uuid.uuid1()}.jpg"), image.numpy())
        counter += data_augmentation_factor
    print(
        f"Successfully generated {counter} new images through data augmentation.")


###################################################################################################
#   DATA & PIPELINE LOADER


def build_dataloader_pipeline(data, max_data_size=100):
    TRAINING_TESTING_RATIO = .7
    # Build dataloader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)
    data = data.take(max_data_size)

    # Training partition
    train_data = data.take(round(len(data)*TRAINING_TESTING_RATIO))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # Testing partition
    test_data = data.skip(round(len(data)*TRAINING_TESTING_RATIO))
    test_data = test_data.take(round(len(data)*(1-TRAINING_TESTING_RATIO)))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)
    return train_data, test_data

###################################################################################################
#   MACHINE LEARNING DEFINITION


def make_embedding():
    """ Create embedding model. """
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


class L1Dist(Layer):
    # Siamese L1 Distance class

    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding_model):
    """ Create the siamese model from the given embedding model. """
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding_model(input_image), embedding_model(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


def define_train_step(siamese_model):
    LEARNING_RATE = 1e-4
    LOSS_FUNCTION = tf.losses.BinaryCrossentropy()  # Loss
    OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)  # Optimizer

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            X = batch[:2]  # anchor and positive/negative image
            y = batch[2]  # label
            yhat = siamese_model(X, training=True)  # forward pass
            loss = LOSS_FUNCTION(y, yhat)

        grad = tape.gradient(loss, siamese_model.trainable_variables)

        # calculate updated weights and apply to siamese model
        OPTIMIZER.apply_gradients(zip(grad, siamese_model.trainable_variables))
        return loss

    return train_step


def define_training_loop(siamese_model, train_step):
    def train(data, n_epoch):
        for epoch in range(1, n_epoch+1):
            print(f"Epoch {epoch}/{n_epoch}")
            progbar = tf.keras.utils.Progbar(len(data))

            #   Creating metric objects
            r = Recall()
            p = Precision()

            for idx, batch in enumerate(data):  # Loop through each batch
                loss = train_step(batch)  # train step
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx+1)
            print(
                f"Loss : {loss.numpy()}\tRecall : {r.result().numpy()}\tPrecision : {p.result().numpy()}")
    return train


def save_model(model, file_name):
    model_path = os.path.join(SAVED_MODEL_PATH, f"{file_name}.h5")
    model.save(model_path)
    print(f"Successfully saved model at {model_path}.")


def make_embedding_from_pretrained(file_name='embedding_kernel'):
    embedding = make_embedding()
    embedding.load_weights(os.path.join(SAVED_MODEL_PATH, f"{file_name}.h5"))
    return embedding


###################################################################################################
#   MACHINE LEARNING MODEL PERFORMANCE


def measure_performances(model, test_data):
    r = Recall()
    p = Precision()
    auc = AUC()
    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
        auc.update_state(y_true, yhat)

    print(f"Recall : {r.result().numpy()}")
    print(f"Precision : {p.result().numpy()}")
    print(f"AUC : {auc.result().numpy()}")
    return r, p, auc


###################################################################################################
#   MACHINE LEARNING KERNEL TRAINING

def plot(loss, recall, precision):
    assert len(loss) == len(recall) == len(precision)
    N = np.array(range(len(loss))) + 1
    plt.figure()
    plt.plot(N, loss, label='loss')
    plt.plot(N, recall, label='recall')
    plt.plot(N, precision, label='precision')
    plt.show()


###################################################################################################
#   MACHINE LEARNING PREDICTION

def get_verify_result(model, anc_path = ANC_PATH, ver_path = VER_PATH, max_data_size=100):
    data_number = min(len(os.listdir(anc_path)), len(os.listdir(ver_path)))
    anchor_file_paths = tf.data.Dataset.list_files(anc_path+'\*.jpg').take(data_number)
    verify_file_paths = tf.data.Dataset.list_files(ver_path+'\*.jpg').take(data_number)
    print(f"Verifying {data_number} data.")
    data = tf.data.Dataset.zip((anchor_file_paths, verify_file_paths, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor_file_paths)))))
    print(f"data : {len(data)}")
    data = data.map(preprocess_twin)
    print(f"data : {len(data)}")
    data = data.cache()
    print(f"data : {len(data)}")
    data = data.shuffle(buffer_size=10000)
    print(f"data : {len(data)}")
    data = data.take(max_data_size)
    print(f"data : {len(data)}")
    # data = data.batch(16)
    print(f"data : {len(data)}")
    # data = data.prefetch(8)
    print(f"data : {len(data)}")
    test_input, test_val, _ = data.as_numpy_iterator().next()
    print(f"data : {len(data)}")
    results = model.predict([test_input, test_val])
    return results

def verify(model, ver_file_path, anc_path = ANC_PATH, max_data_size=100, verbose=True):
    data_number = min(len(os.listdir(anc_path)), max_data_size)
    anchor_file_paths = tf.data.Dataset.list_files(anc_path+'\*.jpg').take(data_number)
    verify_file_paths = tf.data.Dataset.from_tensor_slices([ver_file_path]*data_number).take(data_number)

    print(f"Verifying {data_number} data.") if verbose else None
    data = tf.data.Dataset.zip((anchor_file_paths, verify_file_paths, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor_file_paths)))))
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)
    data = data.batch(16)
    data = data.prefetch(8)
    test_input, test_val, _ = data.as_numpy_iterator().next()
    results = model.predict([test_input, test_val])
    return np.quantile(results, 0.75)


def get_histogram_scores(model, anc_path = BANC_PATH, pos_path = BPOS_PATH, neg_path = NEG_PATH, max_harvest = 530, n_bins = 30, force_computation=False, data_path=r'gross_scores_naive.npz', xmin=0, xmax=1, ymin=0, ymax=5):
    if force_computation:
        gross_scores_positive = []
        
        for file_name in tqdm.tqdm(os.listdir(os.path.join(pos_path))[:max_harvest]):
            img_path = os.path.join(pos_path, file_name)
            gross_scores_positive.append(verify(model, img_path, anc_path, max_data_size=max_harvest, verbose=False))

        gross_scores_negative = []

        for file_name in tqdm.tqdm(os.listdir(os.path.join(neg_path))[:max_harvest]):
            img_path = os.path.join(neg_path, file_name)
            gross_scores_negative.append(verify(model, img_path, anc_path, max_data_size=max_harvest, verbose=False))

        np.savez(data_path, x=gross_scores_positive, y=gross_scores_negative)
    else:
        npzfile = np.load(data_path)

        gross_scores_positive=npzfile['x']
        gross_scores_negative=npzfile['y']

    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True, dpi = 100)
    
    axs[1].set_xlabel('infered value')

    axs[0].hist(gross_scores_positive, bins=n_bins, zorder=20, label="positive samples", color='tab:green')
    axs[1].hist(gross_scores_negative, bins=n_bins, zorder=20, label="negative samples", color='tab:orange')

    for ax,score in [(axs[0], gross_scores_positive), (axs[1], gross_scores_negative)]:
        q1, q2, q3 = np.quantile(score, 0.25), np.median(score), np.quantile(score, 0.75)
        if (ax==axs[0]):
            ax.plot([q1, q1], [ymin, 2*ymax], label="q1", linestyle='dotted', zorder=100, color='tab:olive')
            ax.plot([q2, q2], [ymin, 2*ymax], label="q2", linestyle='dotted', zorder=100, color='tab:orange')
            ax.plot([q3, q3], [ymin, 2*ymax], label="q3", linestyle='dotted', zorder=100, color='tab:red')
        else:
            ax.plot([q1, q1], [ymin, 2*ymax], label="q1", linestyle='dotted', zorder=100, color='tab:cyan')
            ax.plot([q2, q2], [ymin, 2*ymax], label="q2", linestyle='dotted', zorder=100, color='tab:blue')
            ax.plot([q3, q3], [ymin, 2*ymax], label="q3", linestyle='dotted', zorder=100, color='tab:purple')

        ax.set_ylabel('number of sample')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax.minorticks_on()
        ax.legend(loc='upper left')
        ax.set_ylabel('number of sample')
    
    plt.show()

    return gross_scores_positive, gross_scores_negative

def get_confusion_matrix(data_path, threshold = 0.5):
    npzfile = np.load(data_path)
    gross_scores_positive=npzfile['x']
    gross_scores_negative=npzfile['y']
    true_positive = (gross_scores_positive > threshold).sum()/len(gross_scores_positive)
    true_negative = (gross_scores_negative < threshold).sum()/len(gross_scores_negative)
    false_positive = (gross_scores_negative > threshold).sum()/len(gross_scores_negative)
    false_negative = (gross_scores_positive < threshold).sum()/len(gross_scores_positive)
    return [[true_positive, false_positive], [false_negative, true_negative]]

def confusion_under_threshold(data_path, min=0, max=1, n_bit=100):
    npzfile = np.load(data_path)
    gross_scores_positive=npzfile['x']
    gross_scores_negative=npzfile['y']
    
    thresholds = np.linspace(min, max, n_bit)
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    for threshold in thresholds:
        true_positives.append((gross_scores_positive > threshold).sum()/len(gross_scores_positive))
        true_negatives.append((gross_scores_negative < threshold).sum()/len(gross_scores_negative))
        false_positives.append((gross_scores_negative > threshold).sum()/len(gross_scores_negative))
        false_negatives.append((gross_scores_positive < threshold).sum()/len(gross_scores_positive))
    return np.array(thresholds), [[np.array(true_positives), np.array(false_positives)], [np.array(false_negatives), np.array(true_negatives)]]

def plot_confusion_under_threshold(data_path, min=0, max=1, n_bit=100):
    thresholds, [[true_positives, false_positives], [false_negatives, true_negatives]] = confusion_under_threshold(data_path, min, max, n_bit)
    plt.figure()
    plt.plot(thresholds, true_positives, label="true positives")
    plt.plot(thresholds, false_positives, label="false positives")
    plt.plot(thresholds, false_negatives, label="false negatives", alpha=0.2)
    plt.plot(thresholds, true_negatives, label="true negatives", alpha=0.2)
    plt.plot(thresholds, (true_positives)/(true_positives+false_positives), label="precision")
    plt.plot(thresholds, (true_positives)/(false_negatives+true_positives), label="recall")
    plt.xlabel('threshold')
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.minorticks_on()
    plt.legend(loc='center right')
    plt.show()


def plot_roc(data_path_list, min=0, max=1, n_bit=100):
    cols = ['tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:green', 'tab:green', 'tab:red', 'tab:red', 'tab:cyan', 'tab:cyan', 'tab:purple', 'tab:purple']
    k = -1
    plt.figure(figsize=(6,6))
    for data_path in data_path_list:
        k += 1
        thresholds, [[true_positives, false_positives], [false_negatives, true_negatives]] = confusion_under_threshold(data_path, min, max, n_bit)
        dlabel = str(data_path).split(".")[0].split("_")[2:]
        dlabel = " ".join(dlabel)
        col = cols[k % len(cols)]
        ls = 'solid' if k%2!=0 else 'dashed'
        plt.plot(false_positives, true_positives, label=dlabel, color=col, linestyle=ls)
    plt.plot([0,1], [0,1], color='black', linestyle='dashdot')
    plt.xlabel('false positive')
    plt.ylabel('true positive')
    plt.legend(loc='lower right')
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.minorticks_on()
    plt.show()