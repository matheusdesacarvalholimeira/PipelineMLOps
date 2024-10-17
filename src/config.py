import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    MODEL_DIR = os.getenv("MODEL_DIR")
    DATA_DIR = os.getenv("DATA_DIR")
    TRAINING_DATA = os.path.join(DATA_DIR, 'processed', 'training_data.csv')
    TEST_DATA = os.path.join(DATA_DIR, 'processed', 'test_data.csv')