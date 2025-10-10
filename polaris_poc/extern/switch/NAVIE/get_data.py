from elasticsearch import Elasticsearch
import csv
import json
from Custom_Logger import logger
# Connect to Elasticsearch
es = Elasticsearch(['http://localhost:9200'])

# Define the index name and query
def write_csv(index_name , csv_file):

    try:
        query = {"query": {"match_all": {}}}

        # Execute the search request
        response = es.search(index=index_name, body=query, size=10000)  # Adjust the size as needed

        # Extract relevant data from the response
        hits = response['hits']['hits']
        data = [hit['_source'] for hit in hits]

        # Define the CSV file path
        
        # Write the data to a CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print('Data exported to CSV:', csv_file)
    except Exception as e:
        strr = f"Error While downloading metrics data from ES {str(e)}"
        logger.error(strr)
        print(strr) 
        return 

def write_json(index_name , json_file):
    try:
        query = {"query": {"match_all": {}}}

        # Execute the search request
        response = es.search(index=index_name, body=query, size=10000)  # Adjust the size as needed

        # Extract relevant data from the response
        hits = response['hits']['hits']
        data = [hit['_source'] for hit in hits]
        
        # Write the data to a JSON file
        with open(json_file, 'w') as file:
            json.dump(data, file, indent=2)

        print('Data exported to JSON:', json_file)
    except Exception as e:
        strr = f"Error While downloading logs data from ES {str(e)}"
        logger.error(strr)
        print(strr) 
        return 

#saves metrics data to csv
write_csv('final_metrics_data' , 'exported-data-metrics.csv')
#saves logs data to json file
write_json('new_logs' , 'exported-data-logs.json')


