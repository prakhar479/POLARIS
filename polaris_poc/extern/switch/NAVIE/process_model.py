from ultralytics import YOLO
if __name__ == '__main__':

    models = {}
    for m in {'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'}:
        #models[m] = torch.hub.load('ultralytics/ultralytics', m, force_reload=False, device='cpu')
        z = m + ".pt"
        models[m] = YOLO(z)
    # logger.info(    {'Component': "Process" , "Action": "Model's loaded ready to start processing" }  ) 

    print("Model Loaded")
