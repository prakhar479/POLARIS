import os
import json
import random
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO

# ------------------ CONFIG ------------------
class Config:
    """Groups all configuration settings."""
    COCO_YEAR = "2017"
    SAMPLE_SIZE = 2000
    
    # Base URLs
    BASE_URL = "http://images.cocodataset.org"
    ANNOTATIONS_URL = f"{BASE_URL}/annotations/annotations_trainval{COCO_YEAR}.zip"
    IMAGES_BASE_URL = f"{BASE_URL}/train{COCO_YEAR}/"

    # Data directories and file paths
    # Using pathlib for robust path handling
    DATA_DIR = Path("data/coco_sample")
    ANNOTATIONS_DIR = DATA_DIR / "annotations"
    IMAGES_DIR = DATA_DIR / "images"
    
    # Path to the downloaded zip file and the final annotation file
    ANNOTATIONS_ZIP_PATH = ANNOTATIONS_DIR / f"annotations_trainval{COCO_YEAR}.zip"
    ANNOTATIONS_FILE_PATH = ANNOTATIONS_DIR / "annotations" / f"instances_train{COCO_YEAR}.json"
    
    # File to store the list of sampled image IDs for resumability
    SAMPLE_IDS_FILE = DATA_DIR / "sample_ids.json"
# --------------------------------------------

class CocoDownloader:
    """
    A class to download a random sample of the COCO dataset.
    
    Designed to be resumable and robust.
    """
    def __init__(self, config):
        self.config = config
        self.coco = None

    def _ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.config.ANNOTATIONS_DIR.mkdir(exist_ok=True)
        self.config.IMAGES_DIR.mkdir(exist_ok=True)

    def _download_annotations(self):
        """
        Downloads and extracts the COCO annotations if they are not already present.
        """
        if self.config.ANNOTATIONS_FILE_PATH.exists():
            print("✅ Annotation file found. Skipping download.")
            return

        # Step 1: Download the zip file if it doesn't exist
        if not self.config.ANNOTATIONS_ZIP_PATH.exists():
            print(f"Downloading COCO annotations to {self.config.ANNOTATIONS_ZIP_PATH}...")
            try:
                with requests.get(self.config.ANNOTATIONS_URL, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with open(self.config.ANNOTATIONS_ZIP_PATH, "wb") as f, tqdm(
                        desc="Downloading annotations",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            bar.update(size)
            except requests.exceptions.RequestException as e:
                print(f"❌ Error downloading annotations: {e}")
                # Clean up partially downloaded file
                if self.config.ANNOTATIONS_ZIP_PATH.exists():
                    os.remove(self.config.ANNOTATIONS_ZIP_PATH)
                exit(1)

        # Step 2: Extract the zip file
        print(f"Extracting annotations from {self.config.ANNOTATIONS_ZIP_PATH}...")
        with zipfile.ZipFile(self.config.ANNOTATIONS_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(self.config.ANNOTATIONS_DIR)
        print("✅ Annotations extracted.")

    def _load_coco_annotations(self):
        """Loads the COCO API object from the annotation file."""
        print("Loading COCO annotations into memory...")
        self.coco = COCO(str(self.config.ANNOTATIONS_FILE_PATH))

    def _get_or_create_sample_ids(self):
        """
        Loads a previously saved list of sample IDs or creates a new one.
        This is the key to making the download process resumable.
        """
        if self.config.SAMPLE_IDS_FILE.exists():
            print(f"Found existing sample list at {self.config.SAMPLE_IDS_FILE}. Resuming download.")
            with open(self.config.SAMPLE_IDS_FILE, 'r') as f:
                return json.load(f)
        else:
            print("No previous sample found. Generating a new random sample.")
            all_img_ids = self.coco.getImgIds()
            print(f"Total images available in dataset: {len(all_img_ids)}")
            sample_ids = random.sample(all_img_ids, self.config.SAMPLE_SIZE)
            with open(self.config.SAMPLE_IDS_FILE, 'w') as f:
                json.dump(sample_ids, f)
            print(f"Saved new sample list to {self.config.SAMPLE_IDS_FILE}.")
            return sample_ids

    def _download_image(self, img_info):
        """Downloads a single image, skipping if it already exists."""
        file_name = img_info['file_name']
        save_path = self.config.IMAGES_DIR / file_name

        if save_path.exists():
            return False  # False indicates that no new download happened

        img_url = self.config.IMAGES_BASE_URL + file_name
        try:
            resp = requests.get(img_url, stream=True, timeout=15)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True # True indicates a successful new download
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error downloading {img_url}: {e}")
            return False

    def run(self):
        """Executes the entire download and sampling process."""
        self._ensure_directories_exist()
        self._download_annotations()
        self._load_coco_annotations()

        sample_ids = self._get_or_create_sample_ids()
        img_infos = self.coco.loadImgs(sample_ids)

        print(f"\nStarting image download for {self.config.SAMPLE_SIZE} samples to {self.config.IMAGES_DIR}/")
        
        newly_downloaded = 0
        for img_info in tqdm(img_infos, desc="Downloading images"):
            success = self._download_image(img_info)
            if success:
                newly_downloaded += 1
        
        print("\n--- Download Summary ---")
        print(f"✅ Process complete.")
        print(f" Newly downloaded images this session: {newly_downloaded}")
        
        # Final count of images in the directory
        existing_images = len(list(self.config.IMAGES_DIR.glob('*.jpg')))
        print(f" Total images in directory: {existing_images}/{self.config.SAMPLE_SIZE}")

if __name__ == "__main__":
    config = Config()
    downloader = CocoDownloader(config)
    downloader.run()