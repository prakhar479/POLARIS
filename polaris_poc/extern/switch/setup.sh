#!/bin/bash

echo "Importing Dashboard"
curl -X POST 'http://localhost:5601/api/saved_objects/_import' -H 'kbn-xsrf: true' --form file=@export.ndjson

# Setting up Frontend
echo "Setting up Frontend"
echo "Installing Node modules..."
npm install

# Setting up Backend: Model loader, MAPE-K, Locust a load tester
echo "Setting up Backend"
echo "Changing directory to NAVIE..."
cd NAVIE


echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Loading Model's"
python3 process_model.py
