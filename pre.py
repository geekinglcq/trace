import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from pandas import DataFrame, Series


def handle_one(line):
    line = line.strip().split(' ')
    try:
        ID = int(line[0])
        dots = [i.split(',') for i in line[1].strip().split(';')]
        dx,dy = []
    except IndexError as e:
        print(line,e)
        return None



