import os
import shutil

def cleanTempDir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)