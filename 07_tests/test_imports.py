print("Starting import test")
import os
print("os imported")
import json
print("json imported")
import torch
print("torch imported")
import torch.nn as nn
print("torch.nn imported")
import cv2
print("cv2 imported")
import numpy as np
print("numpy imported")
from config_loader import load_config
print("config_loader imported")
config = load_config()
print("config loaded")
from i3d import InceptionI3d
print("i3d imported")
print("All imports successful")
