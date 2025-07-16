"""This module is to train a given model"""

from train_test_utilities import *
from data_processing import *
from logger_utils import create_logger

def train():
    logger = create_logger("")
    validate_rna_sequences("",0,100,logger)