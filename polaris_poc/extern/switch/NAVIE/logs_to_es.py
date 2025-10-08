import re
import json
from datetime import datetime
from elasticsearch import Elasticsearch
import time
from Custom_Logger import logger

# Configure Elasticsearch connection
es = Elasticsearch(['localhost'])  # Replace with your Elasticsearch host

log_file_path = 'logs/Object_detection.log'  # Replace with the actual path to your log file

# Regular expression pattern to extract log data
log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (INFO|DATA) (.*)')
index_name = "new_logs"

# Function to create index if not present
def create_index_if_not_exists(index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)

# Function to convert log data to JSON format
def parse_log_data(line):
    try: 
        match = log_pattern.match(line)
        if match:
            timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
            log_level = match.group(2)
            log_data = match.group(3)

            try:
                log_data = json.loads(log_data.replace("'", '"'))
            except json.JSONDecodeError:
                log_data = {}

            return {
                'timestamp': timestamp,
                'log_level': log_level,
                **log_data  # Unpack log_data dictionary as separate key-value pairs
            }
    except Exception as e:
        error_msg = f"Failed to upload logs data to ES: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        
    return None

# Function to clear log file
def clear_log_file():
    with open(log_file_path, 'w'):
        pass

# Create the index pattern if not present
create_index_if_not_exists(index_name)

# Delete all documents in the index
es.delete_by_query(index=index_name, body={"query": {"match_all": {}}})

# Continuously read log file and send data to Elasticsearch
while True:
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

        if lines:
            for line in lines:
                log_entry = parse_log_data(line)
                if log_entry:
                    print(log_entry)
                    es.index(index=index_name, body=log_entry)

            clear_log_file()

    # Wait for a specific duration before checking the log file again
    # Adjust the sleep duration based on your requirements
    time.sleep(5)  # Wait for 5 seconds before checking the log file again

