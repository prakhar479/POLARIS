import argparse
import time
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import csv
from Custom_Logger import logger
import os
import shutil

# Creating a FastAPI endpoint.
app = FastAPI()
input_rate = 0
start_time = 0
total_in = 0



DETECTION_URL = '/v1/object-detection'


@app.post(DETECTION_URL)
async def predict( image: UploadFile = File(...) ):
    global start_time
    global input_rate
    global total_in
    try: 
        if(time.time() - start_time > 1 ):
                f = open("monitor.csv", "w")
                f.write(f'{input_rate}')
                f.close()
                start_time = time.time()
                input_rate = 0
             
        input_rate+=1
        im_bytes = await image.read()
        x = time.time()
        filename = f"images/queue{total_in}.csv"

        f = open(filename, "w")
        writer = csv.writer(f)
        writer.writerow([x])
        writer.writerow(im_bytes)
        f.close()
        
        total_in += 1
        return 
    except Exception as e:
        strr = str(e)
        print(strr)


if __name__ == '__main__':
    port = 5000
    parser = argparse.ArgumentParser(description='Fast API exposing YOLOv5 model')
    parser.add_argument('--port', default=port, type=int, help='port number')
    opt = parser.parse_args()

    # folder_path = "images"

    # # Check if the folder exists
    # if os.path.exists(folder_path):
    #     # If it exists, remove its contents (files and subdirectories)
    #     for item in os.listdir(folder_path):
    #         item_path = os.path.join(folder_path, item)
    #         if os.path.isfile(item_path):
    #             os.remove(item_path)
    #         elif os.path.isdir(item_path):
    #             shutil.rmtree(item_path)
    # else:
    #     # If it doesn't exist, create the folder
    #     os.mkdir(folder_path)


    uvicorn.run(app, host='0.0.0.0', port=opt.port)
