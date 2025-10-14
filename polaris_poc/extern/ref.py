from Analyzer import Analyzer
import time
import pandas as pd
from elasticsearch import Elasticsearch
import sys

analyzer_obj = Analyzer()
es = Elasticsearch(['http://localhost:9200'])
index_name = 'final_metrics_data'

def get_pending_images_count():
    doc_count = es.count(index="image_data")["count"]
    return doc_count

def get_past_50_rows_average():
    # Define the fields for which you want to calculate the mean
    # fields = ["model_processing_time", "detection_boxes", "confidence"]
    fields = ["image_processing_time",  "confidence", "utility"]
    # Define the number of documents to consider
    num_documents = 10

    # Get the total count of documents in the index
    doc_count = es.count(index=index_name)["count"]

    # Calculate the number of documents to fetch
    num_docs_to_fetch = min(num_documents, doc_count)

    # Set the query to fetch the desired documents
    query = {
        "size": num_docs_to_fetch,
        "sort": [
            {"log_id": {"order": "desc"}}
        ]
    }

    # Fetch the documents from Elasticsearch
    response = es.search(index=index_name, body=query)

    # Initialize dictionaries to store the values for each field
    field_values = {field: [] for field in fields}

    # Extract the field values from the fetched documents
    for hit in response["hits"]["hits"]:
        for field in fields:
            field_value = hit["_source"][field]
            try:
                field_value = float(field_value)
                field_values[field].append(field_value)
            except ValueError:
                pass

    # Calculate the mean for each field
    mean_values = {field: sum(field_values[field]) / len(field_values[field]) if field_values[field] else 0
                for field in fields}

    # Print the mean values
    temp_dict={}

    for field, mean_value in mean_values.items():
        temp_dict[field] = mean_value 
        # print(f"Mean {field}: {mean_value}")
    # fields = ["image_processing_time",  "confidence", "utility"]
    return [temp_dict["image_processing_time"],temp_dict["confidence"],temp_dict["utility"]]

# return [avg_confidence, avg_response_time, avg_detection_boxes]
class Monitor():

    def continous_monitoring(self):
        monitor_dict = {}
        # logger.info(    {'Component': "Monitor" , "Action": "Started the adaptation effector module" }  ) 
        st = time.time()
        while (1):

            if (time.time() - st > 20):
                try:
                    # get the average of past 50 logged data
                    last_50 = get_past_50_rows_average()
                    print(last_50)
                   
                    # retriev current model from model.csv file
                    df = pd.read_csv('../model.csv', header=None)
                    array = df.to_numpy()
                    model_name = array[0][0]
                    monitor_dict["model"] = model_name
                    monitor_dict["image_processing_time"] = last_50["image_processing_time"]
                    monitor_dict["confidence"] = last_50["confidence"]
                    monitor_dict["utility"] = last_50["utility"]
                  
                    print(monitor_dict)
                 
                    if (model_name != 'yolov5n' and model_name != 'yolov5s' and model_name != 'yolov5l' and model_name != 'yolov5m' and model_name != 'yolov5x'):
                        continue

                    analyzer_obj.perform_analysis(monitor_dict)
                    st = time.time()
                    
                except Exception as e:
                    print(e)
                    # logger.error(e)


if __name__ == '__main__':
    monitor_obj = Monitor()
    monitor_obj.continous_monitoring()