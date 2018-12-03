import numpy as np

MAX_NUM_POINT = 4096

def get_dataset(test_area_int=6, num_point=4096):
    assert isinstance(test_area_int, int)
    assert num_point <= MAX_NUM_POINT
