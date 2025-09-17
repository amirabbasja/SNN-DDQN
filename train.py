from models import *
from collections import deque, namedtuple
from huggingface_hub import HfApi, login
import os
import argparse
from utils import *
from IPython.display import clear_output

import sys
import dotenv
from pprint import pprint

from tqdm import tqdm
import pandas as pd
import random, imageio, time, copy
import numpy as np
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

from DDQN import DDQN

# Parse model arguments
parser = modelParamParser()
args, unknown = parser.parse_known_args()


uploadInfo = None
if args.upload_to_cloud:
    # Login to huggingface
    huggingface_read = os.getenv("huggingface_read")
    huggingface_write = os.getenv("huggingface_write")
    repoID = os.getenv("repo_ID")
    api = HfApi()
    login(token = huggingface_write)
    
    uploadInfo = {
        "platform": "huggingface",
        "api": api,
        "repoID": repoID,
        "dirName": "",
        "private": False,
        "replace": True
    }

continueLastRun = args.continue_run
_, runSavePath = get_next_run_number_and_create_folder(continueLastRun, args)

# Copy the config file to the run folder
shutil.copyfile(os.path.join(os.path.dirname(__file__), "conf.json"), os.path.join(runSavePath, "conf.json"))

# Make the environment
env = gym.make("LunarLander-v3")
state, info = env.reset() # Get a sample state of the environment 
stateSize = env.observation_space.shape # Number of variables to define current step 
nActions = env.action_space.n # Number of actions 
actionSpace = np.arange(nActions).tolist() 

# Make the model objects
if args.architecture == "ann":
    qNetwork_model = qNetwork_ANN([stateSize[0], *args.hidden_layers, nActions])
    targetQNetwork_model = qNetwork_ANN([stateSize[0], *args.hidden_layers, nActions])
elif args.architecture == "snn":
    qNetwork_model = qNetwork_SNN([stateSize[0], *args.hidden_layers, nActions], beta = args.snn_beta, tSteps = args.snn_tSteps, DEBUG = args.debug)
    targetQNetwork_model = qNetwork_SNN([stateSize[0], *args.hidden_layers, nActions], beta = args.snn_beta, tSteps = args.snn_tSteps, DEBUG = args.debug)
else:
    raise ValueError(f"Unknown architecture: {args.architecture}")

# Two models should have identical weights initially
targetQNetwork_model.load_state_dict(qNetwork_model.state_dict())

# TODO: Add gradient clipping to the optimizer for avoiding exploding gradients
# Suitable optimizer for gradient descent
optimizer_main = torch.optim.Adam(qNetwork_model.parameters(), lr = args.learning_rate)
optimizer_target = torch.optim.Adam(targetQNetwork_model.parameters(), lr = args.learning_rate)

_networks = {
    "qNetwork_model": qNetwork_model,
    "targetNetwork_model": targetQNetwork_model,
    "optimizer_main": optimizer_main,
    "optimizer_target": optimizer_target
}


args = vars(args) # Convert to dictionary
args["nEpisodes"] = 15000 
args["maxNumTimeSteps"] = 1000
args["action_space"] = actionSpace
args["env"] = env
args["stateSize"] = stateSize

args["memorySize"] = 100000
args["startEbsilon"] = 1.0
args["ebsilonEnd"] = 0.01
args["numUpdateTS"] = 4
args["agents"] = 1  # For now, only 1 agent is supported
args["uploadInfo"] = uploadInfo
args["run_save_path"] = runSavePath

agent = DDQN("HomePC", args, _networks)
agent.train()