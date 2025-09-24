from models import *
from collections import deque, namedtuple
from huggingface_hub import HfApi, login
import os
import argparse
from utils import *
from IPython.display import clear_output

import sys
import dotenv

from tqdm import tqdm
import pandas as pd
import random, imageio, time, copy
import numpy as np
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt

import torch.nn as nn
import torch


# Parse model arguments
parser = modelParamParser()
args, unknown = parser.parse_known_args()

# Load necessary environment variables
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
session_name = os.getenv("session_name")

uploadInfo = None
uploadToCloud = args.upload_to_cloud
local_backup = args.local_backup
lastUploadTime = 0
if uploadToCloud:
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
_, runSavePath = get_next_run_number_and_create_folder(continueLastRun)

# Copy the config file to the run folder
shutil.copyfile(os.path.join(os.path.dirname(__file__), "conf.json"), os.path.join(runSavePath, "conf.json"))

if __name__ == "__main__":
    runStartTime = time.time() # The time the training begun
    maxRunTime = runStartTime + args.max_run_time

    # Making the environment
    NUM_ENVS = args.agents
    env = gym.make("LunarLander-v3") # Use render_mode = "human" to render each episode

    state, info = env.reset() # Get a sample state of the environment
    stateSize = [2] # Number of variables to define current step
    stateSize = env.observation_space.shape # Number of variables to define current step
    nActions = env.action_space.n # Number of actions
    actionSpace = np.arange(nActions).tolist()
    nObs = len(state) # Number of features

    # Set pytorch parameters: The device (CPU or GPU) and data types
    __device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    __dtype = torch.float

    # Model parameters
    projectName = args.name
    hiddenNodes = args.hidden_layers
    learningRate = args.learning_rate
    eDecay = args.decay
    miniBatchSize = args.batch # The length of mini-batch that is used for training
    gamma = args.gamma # The discount factor
    extraInfo = args.extra_info
    continueLastRun = args.continue_run
    debugMode = args.debug
    stopLearningPercent = args.stop_learning_at_win_percent

    # handle the save location
    backUpData = {}
    modelDetails = f"{'_'.join([str(l) for l in hiddenNodes])}_{learningRate}_{eDecay}_{miniBatchSize}_{gamma}_{NUM_ENVS}_{extraInfo}"
    
    # Set the upload directory name
    if uploadToCloud:
        uploadInfo["dirName"] = f"./{session_name}-{projectName}_{modelDetails}"
    
    saveFileName = f"{projectName}_{modelDetails}.pth"
    
    # Make the model objects
    if args.architecture == "ann":
        qNetwork_model = qNetwork_ANN([stateSize[0], *hiddenNodes, nActions]).to(__device, dtype = __dtype)
        targetQNetwork_model = qNetwork_ANN([stateSize[0], *hiddenNodes, nActions]).to(__device, dtype = __dtype)
    elif args.architecture == "snn":
        qNetwork_model = qNetwork_SNN([stateSize[0], *hiddenNodes, nActions], beta = args.snn_beta, tSteps = args.snn_tSteps, DEBUG = debugMode).to(__device, dtype = __dtype)
        targetQNetwork_model = qNetwork_SNN([stateSize[0], *hiddenNodes, nActions], beta = args.snn_beta, tSteps = args.snn_tSteps, DEBUG = debugMode).to(__device, dtype = __dtype)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    # Two models should have identical weights initially
    targetQNetwork_model.load_state_dict(qNetwork_model.state_dict())

    # TODO: Add gradient clipping to the optimizer for avoiding exploding gradients
    # Suitable optimizer for gradient descent
    optimizer_main = torch.optim.Adam(qNetwork_model.parameters(), lr=learningRate)
    optimizer_target = torch.optim.Adam(targetQNetwork_model.parameters(), lr=learningRate)

    # Starting episode and ebsilon
    startEpisode = 0
    startEbsilon = None
    lstHistory = None

    # Making the memory buffer object
    memorySize = 100_000 # The length of the entire memory
    mem = ReplayMemory(memorySize, __dtype, __device)

    if continueLastRun and os.path.isfile(os.path.join(runSavePath, saveFileName)):
        # Load necessary parameters to resume the training from most recent run 
        load_params = {
            "qNetwork_model": qNetwork_model,
            "optimizer_main": optimizer_main,
            "targetQNetwork_model": targetQNetwork_model,
            "trainingParams": [startEpisode, startEbsilon, lstHistory, eDecay, NUM_ENVS, mem]
        }

        # NUM_ENVS is a constant and is defined when running the script for the first time, So we disregard re-loading it
        qNetwork_model, optimizer_main, targetQNetwork_model, startEpisode, startEbsilon, lstHistory, eDecay, _, mem = loadNetwork(os.path.join(runSavePath, saveFileName), **load_params)
        print("Continuing from episode:", startEpisode)

    print(f"Device is: {__device}")

    # Start the timer
    tstart = time.time()

    # The experience of the agent is saved as a named tuple containing various variables
    agentExp = namedtuple("exp", ["state", "action", "reward", "nextState", "done"])

    # Parameters
    nEpisodes = 15000 # Number of learning episodes
    maxNumTimeSteps = 1000 # The number of time step in each episode
    ebsilon = 1 if startEbsilon == None else startEbsilon # The starting  value of ebsilon
    ebsilonEnd  = .01 # The finishing value of ebsilon
    eDecay = eDecay # The rate at which ebsilon decays
    numUpdateTS = 4 # Frequency of time steps to update the NNs
    numP_Average = 100 # The number of previous episodes for calculating the average episode reward

    # Variables for saving the required data for later analysis
    episodePointHist = [] # For saving each episode's point for later demonstration
    episodeHistDf = None
    lstHistory = [] if lstHistory == None else lstHistory
    initialCond = None # Initial condition (state) of the episode
    epPointAvg = -999999 if len(lstHistory) == 0 else pd.DataFrame(lstHistory).iloc[-numP_Average:]["points"].mean()
    latestCheckpoint = 0
    _lastPrintTime = 0
    
    _last100WinPercentage = 0
    for episode in range(startEpisode, nEpisodes):
        initialSeed = random.randint(1,1_000_000_000) # The random seed that determines the episode's I.C.
        state, info = env.reset(seed = initialSeed)
        points = 0
        actionString = ""
        initialCond = state

        tempTime = time.time()

        if debugMode:
            _totalSpikes = 0
            _spikesPerLayer = [0 for _ in range(len(qNetwork_model.layers))]
            _nActionInEpisode = np.array([0 for _ in range(nActions)])

        for t in range(maxNumTimeSteps):
            qValueForActions, trainInfo = qNetwork_model(torch.tensor(state, device = __device, dtype = __dtype).unsqueeze(0))

            # use ebsilon-Greedy algorithm to take the new step
            action = getAction(qValueForActions, ebsilon, actionSpace, __device).cpu().numpy()[0]

            if debugMode:
                _totalSpikes += trainInfo["totalSpikes"]
                _spikesPerLayer = [spikes + newFeedForwardSpikes for spikes, newFeedForwardSpikes in zip(_spikesPerLayer, trainInfo["spikesPerLayer"])]
                _nActionInEpisode[action] += 1
                _gradientNorms, _layerWiseNorms = computeGradientNorms(qNetwork_model)

            # Take a step
            observation, reward, terminated, truncated, info = env.step(action)

            # Store the experience of the current step in an experience deque.
            mem.addNew(agentExp(state, action,reward, observation,True if terminated or truncated else False))

            if not stopLearningPercent < _last100WinPercentage:
                # Check to see if we have to update the networks in the current step
                update = updateNetworks(t, mem, miniBatchSize, numUpdateTS)
                
                if update:
                    # Update the NNs
                    experience = mem.sample(miniBatchSize)

                    # Update the Q-Network and the target Q-Network
                    # Bear in mind that we do not update the target Q-network with direct gradient descent.
                    # so there is no optimizer needed for it
                    fitQNetworks(experience, gamma, [qNetwork_model, optimizer_main], [targetQNetwork_model, None])

            # Save the necessary data
            points += reward
            state = observation.copy()
            actionString += f"{action},"

            # Print the training status. Print only once each second to avoid jitters.
            if 1 < (time.time() - _lastPrintTime):
                os.system('cls' if os.name == 'nt' else 'clear')
                _lastPrintTime = time.time()
                print(f"ElapsedTime: {int(time.time() - tstart): <5}s | Episode: {episode: <5} | Timestep: {t: <5} | The average of the {numP_Average: <5} episodes is: {int(epPointAvg): <5}")
                print(f"Latest chekpoint: {latestCheckpoint} | Speed {t/(time.time()-tempTime+1e-9):.1f} tps | ebsilon: {ebsilon:.3f} ")
                print(f"Remaining time of this run: {maxRunTime - time.time():.1f}s | Remaining time of all trainings: {args.train_finish_timestamp - time.time():.1f}s")
                print(f"Memory details: {mem.len}")
                print("===========================")

            if terminated or truncated or maxRunTime < time.time() or args.train_finish_timestamp < time.time(): break

        _last100WinPercentage = np.sum([1 if exp["finalEpisodeReward"] > 75 else 0 for exp in lstHistory[-100:]]) / 100

        # Save the episode history in dataframe
        lstHistory.append({
            "episode": episode,
            "seed": initialSeed,
            "points": points,
            "timesteps": t,
            "duration": time.time() - tempTime,
            "finalEpisodeReward": reward, # For deducting if termination is for winning or losing
            "state": "terminated" if terminated else "truncated" if truncated else "none", # terminated or truncated or none
            "totalSpikes": _totalSpikes if args.architecture == "snn" and debugMode else None,
            "avgSpikes": _totalSpikes/t if args.architecture == "snn" and debugMode else None,
            "spikesPerLayer": _spikesPerLayer if args.architecture == "snn" and debugMode else None,
            "avgSpikesPerLayer": [spikes/t for spikes in _spikesPerLayer] if args.architecture == "snn" and debugMode else None,
            "nActionInEpisode": _nActionInEpisode,
            "totalGradientNorms": _gradientNorms if debugMode else None,
            "layerWiseNorms": _layerWiseNorms if debugMode else None
        })
        
        # Saving the current episode's points and time
        episodePointHist.append(points)

        # Getting the average of {numP_Average} episodes
        epPointAvg = np.mean(episodePointHist[-numP_Average:])

        # Decay ebsilon
        ebsilon = decayEbsilon(ebsilon, eDecay, ebsilonEnd)

        # Save model weights and parameters periodically (For later use)
        if local_backup:
            if (episode + 1) % 100 == 0 or episode == 2:
                _exp = mem.exportExperience()
                backUpData = {
                    "episode": episode,
                    'qNetwork_state_dict': qNetwork_model.state_dict(),
                    'qNetwork_optimizer_state_dict': optimizer_main.state_dict(),
                    'targetQNetwork_state_dict': targetQNetwork_model.state_dict(),
                    'targetQNetwork_optimizer_state_dict': optimizer_target.state_dict(),
                    'hyperparameters': {"ebsilon": ebsilon, "eDecay": eDecay, "NUM_ENVS": NUM_ENVS},
                    "elapsedTime": int(time.time() - tstart),
                    "train_history": lstHistory,
                    "experiences": {
                        "state": _exp["state"],
                        "action": _exp["action"],
                        "reward": _exp["reward"],
                        "nextState": _exp["nextState"],
                        "done": _exp["done"]
                    }
                }
                # Save the episode number
                latestCheckpoint = episode
                saveModel(backUpData, os.path.join(runSavePath, saveFileName))
        
        # Upload the lightweight progress data to cloud
        if uploadToCloud:
            # Only save the history and with lower frequency. 
            # Re-upload no sooner than every n minute (To avoid being rate-limited)
            if lastUploadTime + 120 < time.time():
                __data = {"train_history": lstHistory, "elapsedTime": int(time.time() - tstart)}
                backUpToCloud(obj = __data, objName = f"{session_name}-{saveFileName}", info = uploadInfo)
                lastUploadTime = time.time()

        # Plot the progress
        if (episode + 1) % 100 == 0 or episode == 2:
            histDf = pd.DataFrame(lstHistory)

            plotEpisodeReward(histDf, os.path.join(runSavePath, f"episode_rewards.png"))

            plotTrainingProcess(histDf, os.path.join(runSavePath, f"training_process.png"))

            plotGradientNorms(histDf, os.path.join(runSavePath, f"gradient_norms.png"))

    env.close()
