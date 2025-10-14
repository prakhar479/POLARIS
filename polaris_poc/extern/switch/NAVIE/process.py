import io
from PIL import Image
import time
import torch
import psutil
import pandas as pd
import filetype
import os
from Custom_Logger import logger
import base64
import datetime
from ultralytics import YOLO
import csv

models = {}
total_processed = 0
global_total_time = 0

metric_file_name = "metrics.csv"


def call_utility(r , C):
    Rmax = 1  # Maximum acceptable response time
    Rmin = 0.1  # Minimum acceptable response time
    Cmax = 1  # Maximum acceptable confidence score
    Cmin = 0.5 # Minimum acceptable confidence score

    # Penalties for exceeding thresholds
    pdv = 1  # Penalty for exceeding the response time threshold
    pev = 1  # Penalty for exceeding the confidence score threshold

    # Weights for the response time and confidence score in the utility function
    we = 0.5 #Confidence
    wd = 0.5 #Response Time

    if r > Rmax:
        Tτ = (Rmax - r) * pdv
    elif Rmin <= r <= Rmax:
        Tτ = r
    else:  # r < Rmin
        Tτ = (r - Rmin) * pdv


    if C > Cmax:
        Eτ = (Cmax - C) * pev
    elif Cmin <= C <= Cmax:
        Eτ = C
    else:  # C < Cmin
        Eτ = (C - Cmin) * pev

    utility_values = we * Eτ + wd * Tτ
    return utility_values

def get_current():
    df = pd.read_csv('model.csv', header=None)
    array = df.to_numpy()
    # print(array[0][0])
    return array[0][0]


def process_row(im_bytes, total_time):

    # run's the object detection on the received data

    global total_processed
    global global_total_time

    kind = filetype.guess(image_path)
    if kind is not None and kind.mime.startswith('image/'):
      image_format = kind.extension
    else:
      image_format = None

    if image_format is None:
        return
    current_model = get_current()

    
    if current_model in models:
        logger.data( {"User Request Time": total_time , 'model': current_model} )
        try:
            if (total_processed == 0):
                global_total_time = time.time()

            im = Image.open(io.BytesIO(im_bytes))
            current_time = time.time()
            results = models[current_model](im)


            current_cpu = psutil.cpu_percent(interval=None)
            total_processed += 1

            # detection = response.pandas().xyxy[0]
            confidences = results[0].boxes.conf.tolist()
            class_list = results[0].boxes.cls.tolist()
            len_conf = len(confidences)
            total_conf = 0
            current_boxes = 0
            for i in range(0, len_conf):
                if confidences[i] >= 0.35 :
                # and class_list[i] == 0:
                    total_conf += confidences[i]
                    current_boxes += 1

            if current_boxes != 0:
                avg_conf = total_conf / current_boxes
            else:
                avg_conf = 0
                
            t = time.time()
            current_time = t - current_time #model processsing time
            total_time = t - total_time # total time take by image to finally output
            absolute_time = t - global_total_time 

            # writes the logs in a log.csv file.
            # print("To write in log file.\n")
            f = open(metric_file_name, "a")
            ts = datetime.datetime.now().isoformat()

            utility = call_utility(total_time , avg_conf )

            f.write(
                f'{ts},{total_processed},{avg_conf},{current_model},{current_cpu},{current_boxes},{current_time},{total_time},{absolute_time},{utility}\n')
            f.close()

            # save_result(response , total_processed-1)

            # return detection.to_json(orient='records')
        
        except Exception as e:
            logger.error(e)
            return {'error': str(e)}
    else:
        return {'error': f'Model {current_model} not found'}
    


def start_processing():

    global total_processed
    while True:
        r = 0
        image_path = f"images/queue{total_processed}.csv"
        image_path_next = f"images/queue{total_processed+1}.csv"

        if os.path.exists(image_path) == False:
            
            logger.error(f"File {total_processed}.csv does not exist")
            if (os.path.exists(image_path_next) == False):
                time.sleep(0.03)
                continue
            else:
                logger.error(f"Skiping a Image file, processing the next")
                total_processed += 1
                image_path = image_path_next

        logger.data({"Processing File" : total_processed  })
        with open(image_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) >= 2:
            try:
                first_row = rows[1]
                total_time = float(rows[0][0])
                print(total_time)
                first_row = [int(x) for x in first_row]
                first_row = bytes(first_row)

                # Process the first row
                process_row(first_row, total_time)
                # Delete the first row from the CSV file
                os.remove(image_path)
                # Do something with the processed row
                logger.data({"Finished Processing File" : total_processed - 1 })

            except Exception as e:
                logger.error(f"Skiping a Image file, processing the next")
                total_processed += 1

        elif len(rows) == 1:
        
            total_processed += 1
        else:
            logger.error("File not exist")
            time.sleep(0.5)
            continue

def create_or_clear_csv(file_path):
    if os.path.exists(file_path):
        # Clear the content of the existing CSV file
        with open(file_path, 'w') as f:
            f.truncate(0)
            # print(f"Cleared content of {file_path}")

import shutil
if __name__ == '__main__':

    models = {}
    for m in {'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'}:
        #models[m] = torch.hub.load('ultralytics/ultralytics', m, force_reload=False, device='cpu')
        z = m + ".pt"
        models[m] = YOLO(z)
    logger.info(    {'Component': "Process" , "Action": "Model's loaded ready to start processing" }  ) 

    print("Model Loaded")
    create_or_clear_csv(metric_file_name)
    # start processing the images.
    
    folder_path = "images"

    # Check if the folder exists
    if os.path.exists(folder_path):
        # If it exists, remove its contents (files and subdirectories)
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        # If it doesn't exist, create the folder
        os.mkdir(folder_path)
    time.sleep(5)
    start_processing()


