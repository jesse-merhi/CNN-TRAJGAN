#!/usr/bin/env python3
"""Configuration file"""
from pathlib import Path

# Github
MODULE_NAME = 'privtrajgen'
GITHUB_URL = "https://github.com/erik-buchholz/PrivTrajGen"


# Directories
BASE_DIR = str(Path(__file__).parent.resolve()) + '/'
TMP_DIR = BASE_DIR + 'tmp/'
LOG_DIR = BASE_DIR + 'logs/'
RESULT_DIR = BASE_DIR + 'results/'
TENSORBOARD_DIR = BASE_DIR + 'tensorboard/'
PARAM_PATH = BASE_DIR + 'parameters/'
PLOT_DIR = BASE_DIR + 'plots/'