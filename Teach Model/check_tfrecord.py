import tensorflow as tf

# TFRecord 파일 경로
tfrecord_file = 'C:/Users/b7115/Desktop/Teach Model/Letters.tfrecord'

# TFRecord 파일에서 데이터 로드
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# 첫 번째 예제만 출력하여 구조 확인
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
