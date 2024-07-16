from ui_loader import compile_ui_to_py
import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import os
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

compile_ui_to_py(os.path.join('ui', 'main.ui'),
                 os.path.join('ui', 'main.py'))
from ui.main import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        # 기본 설정 코드
        super().__init__()
        self.setupUi(self)  
        self.pushButton.clicked.connect(self.Push)
        self.path = ""

    def Push(self):
        fname = QFileDialog.getOpenFileName(self)
        self.path = fname[0]
        if self.path == "":
            self.textBrowser.append("이미지가 없음")
            return
        if not self.path.endswith(('.png', '.jpg')):
            self.textBrowser.append("이미지 파일이 아님")
            return
        pixmap = QPixmap()
        pixmap.load(self.path)
        scaled_pixmap = pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = load_model("model/keras_model.h5", compile=False)

        # Load the labels
        class_names = open("model/labels.txt", "r").readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(self.path).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

        self.textBrowser.append(f"{class_name[2: ]}")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())