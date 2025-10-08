from locust import HttpUser, task
from gevent import spawn
import csv
import time
import os
import os.path


class MyUser(HttpUser):
    wait_times = []
    n = 0
    image_data = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Read environment variables for folder and CSV file
        IMAGES_FOLDER = os.environ.get('IMAGES_FOLDER')
        filename = os.environ.get('CSV_FILE')

        # Read CSV file containing time intervals
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.wait_times = [float(row[0]) for row in reader]

        # self.n = 0

        # Read image files from the specified folder
        for filename in os.listdir(IMAGES_FOLDER):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_path = os.path.join(IMAGES_FOLDER, filename)
                self.image_data.append(image_path)

    @task
    def my_task(self):
        if self.n >= len(self.wait_times):
            # All rows completed, raise an exception to stop the execution
            self.environment.runner.quit()
            return

        image_file = open(self.image_data[self.n], "rb")
        files = {'image': image_file}

        spawn(self.client.post, "/object-detection", files=files)

        time.sleep(self.wait_times[self.n])

        self.n += 1
