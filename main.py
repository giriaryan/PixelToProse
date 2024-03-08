import sys
import re
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PIL import Image
import torch
from moondream import Moondream
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer
import pyttsx3

class TTSWorker(QThread):
    tts_done = pyqtSignal()

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.engine = None

    def run(self):
        self.engine = pyttsx3.init()
        self.engine.say(self.text)
        self.engine.runAndWait()
        self.tts_done.emit()

    def stop(self):
        if self.engine is not None:
            self.engine.stop()


class ImageDescriberApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel to Prose")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.setStyleSheet("QMainWindow {background-color: #333333; color: #ffffff;}"
                           "QPushButton {background-color: #555555; color: #ffffff; border: none; font-size: 24pt;}"
                           "QPushButton:hover {background-color: #666666;}"
                           "QTextEdit {background-color: #555555; color: #ffffff; border: none;}"
                           "QTextEdit:hover {background-color: #666666;}")

        # Image segment
        self.image_segment = QWidget(self)
        self.image_segment_layout = QHBoxLayout(self.image_segment)
        self.image_segment_layout.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_segment)

        # Upload button
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedHeight(200)
        self.upload_button.setStyleSheet("font-weight: 500;")
        self.layout.addWidget(self.upload_button)

        # Result label and volume button
        self.result_widget = QWidget(self)
        self.result_layout = QHBoxLayout(self.result_widget)
        self.layout.addWidget(self.result_widget)

        self.result_label = QTextEdit(self.result_widget)
        self.result_label.setReadOnly(True)
        self.result_label.setStyleSheet("font-size: 24pt;")
        self.result_layout.addWidget(self.result_label, 2)

        self.volume_button = QPushButton(self.result_widget)
        self.volume_button.setIcon(QIcon("volume_icon.png")) # Provide the path to your volume icon
        self.volume_button.setIconSize(QSize(200, 400))  # Set the size of the icon
        self.volume_button.clicked.connect(self.speak_answer)
        self.result_layout.addWidget(self.volume_button)

        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 18pt; color: white;")
        self.layout.addWidget(self.timer_label, 0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer_started = False
        self.elapsed_seconds = 0

        self.model_id = "vikhyatk/moondream1"
        self.tokenizer = Tokenizer.from_pretrained(self.model_id)
        self.moondream = Moondream.from_pretrained(self.model_id).to(device=torch.device("cpu"))
        self.moondream.eval()
        self.tts_worker = None

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((400, 400))
            image = image.convert('RGB')
            q_image = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Create QLabel to display the image thumbnail
            if not hasattr(self, 'image_label'):
                self.image_label = QLabel(self.image_segment)
                self.image_label.setAlignment(Qt.AlignCenter)
                self.image_segment_layout.addWidget(self.image_label)

            self.image_label.setPixmap(pixmap)
            self.process_image_description(image)

    def process_image_description(self, image):
        if not self.timer_started:
            self.timer.start(1000)
            self.timer_started = True

        self.worker = ImageDescriptionWorker(image, self.tokenizer, self.moondream)
        self.worker.image_description_done.connect(self.display_image_description)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.process)
        self.thread.start()

    def display_image_description(self, description):
        self.timer.stop()
        self.timer_started = False
        self.result_label.setText(f"Output: \n{description}")
        self.result_label.setFont(QFont("Arial", 24))

    def update_timer(self):
        if self.timer_started:
            self.elapsed_seconds += 1
            self.timer_label.setText(f"Elapsed Time: {self.elapsed_seconds} seconds")

    def speak_answer(self):
        answer_text = self.result_label.toPlainText()
        if self.tts_worker and self.tts_worker.isRunning():
            return
        self.tts_worker = TTSWorker(answer_text)
        self.tts_worker.tts_done.connect(self.on_tts_done)
        self.tts_worker.start()
        self.volume_button.setEnabled(False)

    def on_tts_done(self):
        self.tts_worker.quit()
        self.tts_worker.wait()
        self.volume_button.setEnabled(True)


class ImageDescriptionWorker(QThread):
    image_description_done = pyqtSignal(str)

    def __init__(self, image, tokenizer, moondream):
        super().__init__()
        self.image = image
        self.tokenizer = tokenizer
        self.moondream = moondream

    def process(self):
        question = "describe this image in detail"
        image_embeds = self.moondream.encode_image(self.image)

        result_queue = torch.multiprocessing.Queue()
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        answer = self.moondream.answer_question(image_embeds, question, self.tokenizer, streamer=streamer,
                                                result_queue=result_queue)
        buffer = ""
        for new_text in streamer:
            buffer += new_text
            if not new_text.endswith("<") and not new_text.endswith("END"):
                buffer = ""
        filtered_answer = re.sub("<$", "", re.sub("END$", "", buffer))
        predans = filtered_answer.strip() if filtered_answer else answer.strip()

        answer = result_queue.get()
        self.image_description_done.emit(answer)


def main():
    app = QApplication(sys.argv)
    window = ImageDescriberApp()
    window.setGeometry(100, 100, 600, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
