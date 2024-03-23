from re import A
import sys
import os
import math
from datetime import datetime
import random


def create_output_folder(prefix, no_postfix=False):
    folder = prefix
    if not no_postfix:
       folder = folder + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(folder, exist_ok=True)
    return folder


class Tee(object):
    def __init__(self, fn, mode):
        self.file = open(fn, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def rgb2uint(r, g, b):
    return r * 65536 + g * 256 + b
