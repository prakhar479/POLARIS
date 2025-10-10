# SWITCH: An Exemplar for Evaluating Self-Adaptive ML-Enabled Systems

"SWITCH", an exemplar developed to enhance self-adaptive capabilities in Machine Learning-Enabled Systems, through dynamic model SWITCHing in runtime.  "SWITCH" is designed as a comprehensive web service, catering to a broad range of ML scenarios, with its implementation demonstrated through an object detection use case.

## Getting started

The system comprises distinct components, including a frontend written in React, a backend powered by FAST API in Python, and Elasticsearch serving as the database. Will be using YOLOv5u as object detection model .



<details>
<summary><b>Frontend</b></summary>

- **Technology Stack:** React
- **Startup Command:** `npm run start`
- **Port:** 3000
- **Access:** The web application is accessible at http://localhost:3000.
  
</details>

<details>
<summary><b>Backend</b></summary>

- **Technology Stack:** FAST API, Python
- **Startup Command:** `python3 Node.py`
- **Port:** 3001

</details>

<details>
<summary><b>Database</b></summary>

- **Technology:** Elasticsearch and Kibana
- **Data Storage:**
  1. Image data encoded from the input image.
  2. Metrics obtained from object detection, including confidence scores and detection boxes.
  3. System logs.

</details>

#### Requirment's:

- Install [Docker Engine](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) as standalone binaries.
- Minimum free space of 7 GB.
- The system is designed for Linux environments. If you do not have access to a Linux system, you can either utilize a virtual Linux machine or employ a Docker image of a Linux environment.
- If using virtual enviorment for set-up, pre install virtualenv.
  
## Installation

Video for steps of intallation and running the application can be found in:

> Directory: `Video Demonstration`

> Link:
1. Youtube: https://youtu.be/ZIDE1v3jxeQ

2. Google Drive: https://drive.google.com/drive/folders/1eGY1QGXpP4FYkav2G2uusmV6QSHVnoVx?usp=sharing  

If using a zip file for installation:

> Extract the SWITCH folder.

>If not already in the directory `SWITCH` directory, use command from the extracted directory:
```bash
cd SWITCH
```


## Setting up Elastic Search and Kibana 
Start the **docker engine** on your system.

Start Elasticsearch and Kibana containers using Docker Compose with image version 7.9.1, usign command:


```bash
docker-compose up
```

> Wait until ready, this may take time to install depending on the internet connection.
> You can check if it's ready by accessing Elasticsearch at http://localhost:9200/, and Kibana at http://localhost:5601/.

> Leave the docker-container running. Use a different terminal for the rest of the steps.

## Setting up Virtual Environment (optional)

Using virtualenv allows you to avoid installing Python packages globally

```bash
python3 -m venv ./venv

source venv/bin/activate
```

## Installation (only any one is requied)

<details>
<summary><b>Installation using setup.sh</b></summary>


 ```bash
chmod +x setup.sh
./setup.sh
```
</details>

<details>
<summary><b>Manual Installation</b></summary>
 
### Importing Dashboard


```bash
curl -X POST "http://localhost:5601/api/saved_objects/_import" -H "kbn-xsrf: true" --form file=@export.ndjson
```
 
### Setting up Frontend

To install node module's:

```bash
npm install
```


### Setting up Backend: Model loader, MAPE-K, Locust a load tester.


```bash
cd NAVIE

pip install -r requirements.txt

python3 process_model.py

```
</details>

## Run the application

Ensure `docker-compose.yml` is running:

If not running, run following command from directory `SWITCH`
```bash
docker-compose up
```

**If using virtual enviorment**:

Ensure that you are in the virtual enviorment you had created, you can activate virtual enviorment using command:
```
source venv/bin/activate
```


Run the backend first for the application from the directory `SWITCH/NAVIE`:

```
python3 Node.py
```

Run the React-Application from the directory `SWITCH` :

```
npm run start
```


## Using the Application

### If the Image folder size is < 500Mb
1. Upload Image folder in .zip format. The .zip file must have same name as the Image folder.

### If the Image folder size is > 500Mb
1. Enter the full path of your image folder starting from the root directory in the provided text field.
   
### Next steps are same for both the way's of giving Image folder as Input.

2. Create a `.csv` file with inter-arrival rates for images and upload it.
3. Assign an `ID` to your experiment.
4. Choose an approach for running the experiment:
    - 4 self-adaptive approaches.
      - `NAIVE` and `AdaMLs` are different types of provided adaptation techniques.
      - `Modify NAIVE` modifies the NAIVE algorithm according to user input values.
      - `Upload MAPE-K file's` is described below.
    - 5 single model approaches.

5. Click `Upload`.
6. Once the Dashboard is displayed, you can adjust the refresh rate by clicking on the watch icon. A dropdown menu will appear, allowing you to set the desired refresh rate.
7. Stop the process or when all images are processed, click `Stop Process`.
8. Download data for the experiment.
9. You can stat new experiment by clicking `New Process`
  
<details>
<summary><b>Upload MAPE-K file's Guidelines</b></summary>


---

To effectively implement and utilize the MAPE-K framework, follow these steps:

**Upload the Following Files:**

- **`monitor.py`:**
  - Description: This file monitors the relevant metrics for adaptation.
  - Guidelines: Refer to code below to extract nessesary metrics from datastorage.
  - Implementation: Define a planner object and pass the extracted metrics to it.
  

  <details>
  <summary><b>Code for Fetching past n metrics average from Elasticsearch</b></summary>
    
      def fetch_past_n_metrics_average():
    
        fields = ["model_processing_time", "detection_boxes", "confidence"]

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

        # Return the mean values
        temp_dict = {}
        for field, mean_value in mean_values.items():
            temp_dict[field] = mean_value

        return [temp_dict["confidence"], temp_dict["model_processing_time"], temp_dict["detection_boxes"]]
  </details>

  <details>
  <summary><b>Code to Extract model in use and input rate:</b></summary>

      def extract_metric(file_name):
          df = pd.read_csv(file_name, header=None)
          
          array = df.to_numpy()
          return array[0][0]

      monitor_dict = {}  # Initialize the dictionary to store monitored values
      monitor_dict["model"] = extract_metric("../model.csv")
      monitor_dict["Input_rate"] = extract_metric("../monitor.csv")
  </details>
  


 - **`planner.py`:**
   - Description: This code is responsible for determining the necessity of adaptation.
   - Implementation: Develop logic within this file to plan and decide whether adaptation is required.

 - **`Analyzer.py`:**
   - Description: This code determines the result of the adaptation process.
   - Implementation: Include logic to determine the adaptation step.

 - **`Execute.py`:**
   - Description: Executes the model switch.
   - Guidelines: Refer to the code below for model switching.
   - Implementation: Integrate the necessary logic to perform the model switch.

    <details>
    <summary><b>Model switching code</b></summary>

        def switch_model(model_name):
          f = open("../model.csv", "w")
          f.write(model_name)
          f.close()
      
        def perform_action(act):
          # model switch takes place by changing the model name in the model.csv file .
          if (act == 1):
              # switch model to n
              switch_model("YOLOv5n")

          elif (act == 2):
              # switch model to s
              switch_model("YOLOv5s")

          elif (act == 3):
              # switch model to m
              switch_model("YOLOv5m")

          elif (act == 4):
              # switch model to l
              switch_model("YOLOv5l")

          elif (act == 5):
              # switch model to xl
              switch_model("YOLOv5x")

          print("Adaptation completed.")
    </details>


 - **`Knowledge.zip`:**
   - Description: A zip file containing all the knowledge files required by the MAPE-K framework for the successful generation and execution of adaptations.

**Folder Structure:**

   - Your code files are saved in a folder structure: `NAIVE/external_MAPE_K_<id>`.
   - You have the flexibility to make direct changes to any file within this specified directory.

**Note:**

Ensure that the code files adhere to the specified guidelines for seamless integration with the MAPE-K framework.

</details>

## Stoping the Application
Stoping the application, it is necessary to terminate the associated processes executing in each of the three designated terminals. This involves the cessation of the 'Docker-compose' process in the first terminal, the termination of the 'Node.py' script in the second terminal, and the halting of the 'frontend' process in the third terminal. This can be done by pressign `Ctrl+C` in each terminal



## Using the final result's 
If you have downloaded the data for an experiment, the metric's data is stored in a CSV file: `NAVIE/Exported_metrics/exported-data-metrics_{id}` and the logs for the experiment are stored in a JSON file: `NAVIE/Exported_logs/exported-data-logs_{id}`.
 
## Filtering based on classes and confidence score:


This code snippet, to be modified in the `process.py` file located in the `NAVIE` directory, demonstrates the process of filtering object detection results based on class IDs and confidence levels corresponding to the COCO dataset. Users can specify desired classes by adding class filters within the provided loop. Examples for detecting specific classes, such as 'crowd' (class ID 0) and 'dog' (class ID 16), are provided in the comments of the code.

```
# Loop through detected objects and filter based on confidence and class ID
for i in range(0, len_conf):
    # Threshold for detection confidence
    if confidences[i] >= 0.35 : # Additional class filter to be added here
        # Example filters (uncomment as needed):
        # For 'crowd', class ID 0: if class_list[i] == 0:
        # For 'dog', class ID 16: if class_list[i] == 16:
        total_conf += confidences[i]
        current_boxes += 1

# Calculate the average confidence of detected objects
if current_boxes != 0:
    avg_conf = total_conf / current_boxes
else:
    avg_conf = 0
```

## Quick Testing
To get familiar with the tool we provide Data for testing the tool. In the directory `Quick Testing` we have provided a .zip file containing images, and a .csv format Inter arrival rate file for testing purposes. 

## Experimental Result's
The conducted experiments encompassed a range of scenarios, including general object detection, crowd detection, and vehicle detection. For each of these experiments, the corresponding metric files and log files have been compiled and are available in the `Experiments` directory.

The Metrics are present in the `Exported_metrics` directory within the `Experiments` directory. The metrics files are named according to the scenarios they are tested for.

The Logs are present in the `Exported_logs` directory within the `Experiments` directory. The log files are named according to the scenarios they are tested for.

The Images used for Experiment purposes along with inter arrival rate file can be found in the Drive folder:
https://drive.google.com/drive/folders/1MpaJm6-D0xi3xBcf_D_rr5zqsvoSIiro?usp=sharing

## Creating a Inter arrival rate file.
We have also provided a code that scales the `wc_day53-r0-105m-l70.delta` according to the number of Images specified by the user. The code is present in the `Create_rate_file` directory and the code is self-explanatory. 
