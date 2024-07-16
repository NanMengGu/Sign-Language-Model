import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

tfrecord_file = 'C:/Users/b7115/Desktop/Teach Model/Letters.tfrecord'
label_map_file = 'C:/Users/b7115/Desktop/Teach Model/Letters_label_map.pbtxt'
labels_output_file = 'labels.txt'

batch_size = 128
buffer_size = 128
image_size = [224, 224]
num_epochs = 50
learning_rate = 0.0001

def load_label_map(label_map_path):
    from google.protobuf import text_format
    from object_detection.protos import string_int_label_map_pb2

    with open(label_map_path, 'r') as f:
        label_map_string = f.read()
    
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(label_map_string)
    
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.id] = item.display_name
    return label_map_dict

label_map_dict = load_label_map(label_map_file)
num_classes = 26

def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    label = tf.cast(example['image/object/class/label'], tf.int32)
    label = tf.where(label == 26, 25, label)
    return image, label

def load_dataset(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    return parsed_dataset

dataset = load_dataset(tfrecord_file)
dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

model = Sequential([
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):
    model.fit(dataset, epochs=num_epochs)

model.save('trained_model.h5')

def save_label_map(label_map_dict, output_file):
    with open(output_file, 'w') as f:
        for label_id, label_name in label_map_dict.items():
            f.write(f'{label_id}: {label_name}\n')

save_label_map(label_map_dict, labels_output_file)
