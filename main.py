# # для обнаружения объектов на картинках
# from imageai.Detection import ObjectDetection
# # для работы с ос
# import os

# # для ссылки на проект
# exec_path = os.getcwd()
# print("path=", exec_path)
# # Создаём объект класса ObjectDetection
# detector = ObjectDetection()
# # Устанавливаем модель RetinaNet как используемую
# detector.setModelTypeAsRetinaNet()
# # Устанавливаем путь к модели
# detector.setModelPath(os.path.join(
#     exec_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth")
# )
# # Загружаем модель
# detector.loadModel()
# # Список объектов на картинке, которую дадим модели. Путь к данной картинке и создаваемой размеченной
# list = detector.detectObjectsFromImage(
#     input_image=os.path.join(exec_path, "objects.jpg"),
#     output_image_path=os.path.join(exec_path, "new_objects.jpg"),
#     # минимальный процент уверенности в правильности классификации (ниже - объект не классифицирован)
#     minimum_percentage_probability=90,
#     display_percentage_probability=True,
#     display_object_name=False
# )

# Для обнаружения объектов на видео
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()

# Список фреймов из видео
video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    # Число кадров в секунду
    frames_per_second=20,
    log_progress=True)
print(video_path)