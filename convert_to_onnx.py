from ultralytics import YOLO

yolo_model = input("Enter YOLO pt model to convert including extension: ")

model = YOLO(yolo_model)
path = model.export(format="onnx")