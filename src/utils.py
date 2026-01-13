import sys
import os
import dill
import src.logger as logging
from src.exception import CustomException

import pandas as pd
import numpy as np


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        import pickle
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise e