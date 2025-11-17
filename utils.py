from collections import deque, namedtuple
import os
import argparse
import requests
import subprocess
import json
from typing import Union, Dict


from tqdm import tqdm
import pandas as pd
import random, imageio, time, copy
import numpy as np
import gymnasium as gym
import shutil
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen


class ReplayMemory(object):
    """
    Implement's the replay memory algorithm.
    """
    def __init__(self, size, dtype, device) -> None:
        """
        Initialize the class with a double ended queue which will contain named tuples.

        Args:
            size (int): The maximum size of the memory buffer.
            dtype (torch.dtype): The data type of the elements in the memory buffer.
            device (torch.device): The device to store the data (CPU or GPU)
        """
        self.exp = deque([], maxlen=size)
        self.size = size # SIze of the memory
        self.len = len(self.exp)
        self.dtype = dtype
        self.device = device

    def exportExperience(self):
        """
        Exports the experiences to a dictionary of lists.

        Returns:
            Returns a dictionary containing keys "state, action, reward, nextState, done"
        """
        __state = [e.state for e in self.exp]
        __action = [e.action for e in self.exp]
        __reward = [e.reward for e in self.exp]
        __nextState = [e.nextState for e in self.exp]
        __done = [e.done for e in self.exp]

        return {
            "state": __state,
            "action": __action,
            "reward": __reward,
            "nextState": __nextState,
            "done": __done
        }
    
    def loadExperiences(self, state, action, reward, nextState, done):
        """
        Loads previous experiences into the ReplayMemory object.

        Args:   
            Lists of experiences, with the same length
            
        Returns:
            None
        """
        try:
            __experiences = zip(
                reversed(state), 
                reversed(action), 
                reversed(reward), 
                reversed(nextState), 
                reversed(done)
            )
            __tempTuple = namedtuple("exp", ["state", "action", "reward", "nextState", "done"])
            __tempDeque = deque([], maxlen = self.size)

            for __state, __action, __reward, __nextState, __done in __experiences:
                __tempDeque.appendleft(
                    __tempTuple(
                        __state, # Current state
                        __action,
                        __reward, # Current state's reward
                        __nextState, # Next state
                        __done
                    )
                )
            
            self.exp = __tempDeque
            self.len = len(__tempDeque)
        except:
            print("Could not load the data to ReplayMemory object")

    def addNew(self, exp:namedtuple) -> None:
        """
        Adding a new iteration to the memory. Note that the most recent values of
        the training will be located at the position 0 and if the list reaches maxlen
        the oldest data will be dropped.

        Args:
            exp: namedtuple: The experience should be a named tuple with keys named
                like this: ["state", "action", "reward", "nextState", "done"]
        """
        self.exp.appendleft(exp)
        self.len = len(self.exp)

    def addMultiple(self, exps):
        """
        Adds a batch of new experiences to the replay memory. Note that because we
        are using extendLeft here, the experiences will be added in the reverse 
        order: [exp1, exp2, exp3] + [old EXPS]-> [exp3, exp2, exp1, old EXPS]
        
        Args:
            exps: list: A list of named tuples
        """
        self.exp.extendleft(exps)
        self.len = len(self.exp)
    
    def sample(self, miniBatchSize:int, framework = "pytorch") -> tuple:
        """
        Get a random number of experiences from the entire experience memory.
        The memory buffer is a double ended queue (AKA deque) of named tuples. To make
        this list usable for tensor flow neural networks, this each named tuple inside
        the deque has to be unpacked. we use a iterative method to unpack. It may be
        inefficient and maybe using pandas can improve this process. one caveat of using
        pandas tables instead of deque is expensiveness of appending/deleting rows
        (experiences) from the table.

        Args:
            miniBatchSize: int: The size of returned the sample

        Returns:
            A tuple containing state, action, reward, nextState and done
        """
        if framework == "pytorch":
            miniBatch = random.sample(self.exp, miniBatchSize)
            state = torch.from_numpy(np.array([e.state for e in miniBatch if e != None])).to(self.device, dtype = self.dtype)
            action = torch.from_numpy(np.array([e.action for e in miniBatch if e != None])).to(self.device, dtype = torch.int)
            reward = torch.from_numpy(np.array([e.reward for e in miniBatch if e != None])).to(self.device, dtype = self.dtype)
            nextState = torch.from_numpy(np.array([e.nextState for e in miniBatch if e != None])).to(self.device, dtype = self.dtype)
            done = torch.from_numpy(np.array([e.done for e in miniBatch if e != None]).astype(np.uint8)).to(self.device, dtype = torch.int)
        elif framework == "tensorflow":
            raise Exception("TensorFlow is not supported yet")
            # miniBatch = random.sample(self.exp, miniBatchSize)
            # state = tf.convert_to_tensor(np.array([e.state for e in miniBatch if e != None]), dtype=tf.float32)
            # action = tf.convert_to_tensor(np.array([e.action for e in miniBatch if e != None]), dtype=tf.float32)
            # reward = tf.convert_to_tensor(np.array([e.reward for e in miniBatch if e != None]), dtype=tf.float32)
            # nextState = tf.convert_to_tensor(np.array([e.nextState for e in miniBatch if e != None]), dtype=tf.float32)
            # done = tf.convert_to_tensor(np.array([e.done for e in miniBatch if e != None]).astype(np.uint8), dtype=tf.float32)
        return tuple((state, action, reward, nextState, done))

def decayEbsilon(currE: float, rate:float, minE:float) -> float:
    """
    Decreases ebsilon each time called. It multiplies current ebsilon to decrease rate.
    The decreasing is continued until reaching minE.
    """
    return(max(currE*rate, minE))

def computeLoss(experiences:tuple, gamma:float, qNetwork, target_qNetwork):
    """
    Computes the loss between y targets and Q values. For target network, the Q values are
    calculated using Bellman equation. If the reward of current step is R_i, then y = R_i
    if the episode is terminated, if not, y = R_i + gamma * Q_hat(i+1) where gamma is the
    discount factor and Q_hat is the predicted return of the step i+1 with the
    target_qNetwork.

    For the primary Q network, Q values are acquired from the step taken in the episode
    experiences (Not necessarily MAX(Q value)).

    Args:
        experiences (Tuple): A tuple containing experiences as pytorch tensors.
        gamma (float): The discount factor.
        qNetwork (pytorch NN): The neural network for predicting the Q.
        target_qNetwork (pytorch NN): The neural network for predicting the target-Q.

    Returns:
        loss: float: The Mean squared errors (AKA. MSE) of the Qs.
    """
    # Unpack the experience mini-batch
    state, action, reward, nextState, done = experiences

    # with torch.no_grad():
    target_qNetwork.eval()
    qNetwork.eval()

    # To implement the calculation scheme explained in comments, we multiply Qhat by (1-done).
    # If the episode has terminated done == True so (1-done) = 0.
    _targetQValues, _ = target_qNetwork(nextState)
    Qhat = torch.amax(_targetQValues, dim = 1)
    yTarget = reward + gamma *  Qhat * ((1 - done)) # Using the bellman equation

    # IMPORTANT: When getting qValues, we have to account for the ebsilon-greedy algorithm as well.
    # This is why we dont use max(qValues in each state) but instead we use the qValues of the taken
    # action in that step.
    qValues, _ = qNetwork(state)

    qValues = qValues[torch.arange(state.shape[0], dtype = torch.long), action]

    # Calculate the loss
    loss = nn.functional.mse_loss(qValues, yTarget)

    return loss

def fitQNetworks(experience, gamma, qNetwork, target_qNetwork):
    """
    Updates the weights of the neural networks with a custom training loop. The target network is
    updated by a soft update mechanism.

    Args:
        experience (tuple): The data for training networks. This data has to be passed with
            replayMemory.sample() function which returns a tuple of tensorflow tensors in
            the following order: state, action, reward, nextState, done)
        gamma (float): The learning rate.
        qNetwork, target_qNetwork (list): A list of pytorch model and its respective
            optimizer. The first member should be the model, second one its optimizer

    Returns:
        None
    """
    __qNetworkModel = qNetwork[0]
    __qNetworkOptim = qNetwork[1]
    __targetQNetworkModel = target_qNetwork[0]

    # Update the Q network's weights
    loss = computeLoss(experience, gamma, __qNetworkModel, __targetQNetworkModel)

    __qNetworkModel.train()
    __targetQNetworkModel.train()

    __qNetworkOptim.zero_grad()
    loss.backward()
    __qNetworkOptim.step()

    # Update the target Q network's weights using soft updating method
    for targetParams, primaryParams in zip(__targetQNetworkModel.parameters(), __qNetworkModel.parameters()):
        targetParams.data.copy_(targetParams.data * (1 - .001) + primaryParams.data * .001)

def getAction(qVal: list, e:float, actionSpace: list, device: torch.device ) -> int:
    """
    Gets the action via an epsilon-greedy algorithm. This entire action state depends on the env.
    With a probability of epsilon, a random choice will be picked, else the action with
    the greatest Q value will be picked.

    Args:
        qVal: nn.Tensor. The q value of actions of each agent. Axis 0 should be for
            agents.
        e: float: The epsilon which represents the probability of a random action
        actionSpace: list: A list of available actions to take. For example, the 
            available actions possible for LunarLander are as follows:
            [1,2,3,4] => [DoNothing, leftThruster, MainThruster, RightThruster]
        device: torch.device: The device to store the data (CPU or GPU)

    Returns:
        action: int: 0 for doing nothing, and 1 for left thruster, 2 form main thruster
            and 3 for right thruster.
    """
    rndMask = torch.rand(qVal.shape[0], device = device) < e
    actions = torch.zeros(qVal.shape[0], dtype=torch.long, device = device)

    # Get random actions
    actions[rndMask] = torch.randint(0, len(actionSpace), (int(rndMask.sum().item()),), device = device)
    
    # Get optimal actions
    actions[~rndMask] = torch.argmax(qVal[~rndMask], dim = 1)

    return actions

def updateNetworks(timeStep: int, replayMem: ReplayMemory, miniBatchSize: int, C: int) -> bool:
    """
    Determines if the neural network (qNetwork and target_qNetwork) weights are to be updated.
    The update happens C time steps apart. for performance reasons.

    Args:
        timeStep: int: The time step of the current episode
        replayMem: deque: A double edged queue containing the experiences as named tuples.
            the named tuples should be as follows: ["state", "action", "reward", "nextState", "done"]

    Returns:
        A boolean, True for update and False to not update.
    """

    return True if ((timeStep+1) % C == 0 and miniBatchSize < replayMem.len) else False

def getEbsilon(e:float, eDecay:float, minE: float) -> float:
    """
    Decay epsilon for epsilon-Greedy algorithm. epsilon starts with 1 at the beginning of the
    learning process which indicates that the agent completely acts on a random basis (AKA
    Exploration) but as the learning is continued, the rate at which agent acts randomly decreased
    via multiplying the epsilon by a decay rate which ensures agent acting based on it's learnings
    (AKA Exploitation).

    Args:
        e: float: The current rate of epsilon
        eDecay: float: The decay rate of epsilon
        minE: float: the minimum amount of epsilon. To ensure the exploration possibility of the
            agent, epsilon should't be less than a certain amount.

    Returns: epsilon's value
    """

    return max(minE, eDecay * e)

def renderEpisode(initialState: int, actions:str, envName:str, delay:float = .02) -> None:
    """
    Renders the previously done episode so the user can see what happened. We use Gym to
    render the environment. All the render is done in the "human" mode.

    Args:
        initialState: int: The initial seed that determine's the initial state of the episode
            (The state before we took teh first action)
        actions: string: A string of actions delimited by comma (i.e. 1,2,3,1,3, etc.)
        env: string: The name of the environment to render the actions, It has to be a gymnasium
            compatible environment.
        delay: int: The delay (In seconds) to put between showing each step to make it more
            comprehensive.

    Returns: None
    """
    tempEnv = gym.make(envName, render_mode = "human") # Use render_mode = "human" to render each episode
    state, info = tempEnv.reset(seed=initialState) # Get a sample state of the environment

    # Process the string of actions taken
    actions = actions.split(",") # Split the data
    actions = actions[:-1] # Remove the lat Null member of the list
    actions = list(map(int, actions)) # Convert the strings to ints

    # Take steps
    for action in actions:
        _, _, terminated, truncated, _ = tempEnv.step(action)

        # Exit loop if the simulation has ended
        if terminated or truncated:
            _, _ = tempEnv.reset()
            break

        # Delay showing the next step
        time.sleep(delay)

    tempEnv.close()

def analyzeLearning(episodePointHistory:list, episodeTimeHistory:list) -> None:
    """
    Plots the learning performance of the agent

    Args:
        episodePointHistory: list: The commulative rewards of each episode in consrcutive time steps.
        episodeTimeHistory: list: The time it took to run the episode
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    ax1.plot(episodePointHistory)
    ax1.set_title("Episode points")

    ax2.plot(episodeTimeHistory)
    ax2.set_title("Episode elapsed time")

def testAgent(envName:str, network, __device, __dtype, saveVideoName:str = "", ) -> int:
    """
    Runs an agent through a predefined gymnasium environment. The actions of the agent are chosen via
    a greedy policy by a trained neural network. To see the agent in action, the environment's render
    mode has to be "human" or  "rgb-array"

    Args:
        envName: string: The name of the environment.
        network: pytorch NN: The trained neural network that accepts state as an input and outputs
            the desired action.
        environment: gymnasium env: The environment for testing.
        saveVideoName:string: The name of the file to be saved. If equals "", No video file will be
            saved; Also remember that the file name should include the file extension.
    """

    def interactionLoop(env_, seed_, V_):
        """
        The loop that lets agent interact with the environment.
        if V_ == True, save the video (requires render_mode == rgb_array)
        """
        state, _ = env_.reset(seed = seed_)
        points = 0
        if V_:
            videoWriter = imageio.get_writer(saveVideoName)

        maxStepN = 1000
        for t in range(maxStepN):
            # Take greedy steps
            # action = np.argmax(network(np.expand_dims(state, axis = 0)))

            action = torch.argmax(network(torch.tensor(state, device = __device, dtype = __dtype)))

            state, reward, terminated, truncated, _ = env_.step(action.item())

            if V_:
                videoWriter.append_data(env_.render())

            points += reward

            # Exit loop if the simulation has ended
            if terminated or truncated:
                _, _ = env_.reset()

                if V_:
                    videoWriter.close()

                return points

    # Get the random seed to get the initial state of the agent.
    seed = random.randint(0, 1_000_000_000)

    # Because gymnasium doesn't let the environment to have two render modes,
    # we run the simulation twice, The first renders the environment with "human"
    # mode and the second run, runs the environment with "egb_array" mode that
    # lets us save the interaction process to a video file. Both loops are run
    # with the same seeds and neural networks so they should have identical outputs.
    environment = gym.make(envName, render_mode = "human")
    point = interactionLoop(environment, seed, False)

    environment = gym.make(envName, render_mode = "rgb_array")
    point = interactionLoop(environment, seed, True if saveVideoName != "" else False)

    return point

def saveModel(data, location, backup = True, upload = {}):
    """
    Saves the provided data using pytorch's save function. For avoiding 
    data loss, a backup file is created before the data overwritten into 
    the old file (if it already exists). The backup will be saved in the
    child directory of the provided location in the folder "Backup".

    Args:
        data: any: The data to be saved
        location: string: The location to save the data. Should contain the 
            file name.
        backup: bool: If True, a backup of the old file will be created in
            <Backup> directory inside the provided location.
    """
    # First create a backup if the file exists
    if backup:
        if os.path.isfile(location):
            os.makedirs(os.path.join(os.path.dirname(location), "Backups"), exist_ok = True) # Make the "Backups" directory if need be
            shutil.copy2(location, os.path.join(os.path.dirname(location), "Backups")) # Copy the file
            os.remove(location)
    
    # Save the data
    torch.save(data, location)

def backUpToCloud(filePath = None, obj = None, objName = None, info: dict = None) -> Union[str, None]:
    """
    Uploads the data to the cloud. Currently only uses huggingface. Will
    create the repository if it doesn't exist.
    
    Args:
        filePath: string: Path to the file you trying to upload (optional
            if obj is provided).
        obj: object: Python object to save and upload (optional if filePath
            is provided).
        objName: string: Name for the object file (required if obj is provided).
        info: dict: The information about the data and saving location. Example 
        below:
            info = {
                "platform": "huggingface",
                "api": api, # api = HfApi()
                "repoID": f"{your_username}/{your_dataset_name}",
                "dirName": "Data", # The directory to save the data (inside repo)
                "private": False # Upload as a private repo (optional, defaults 
                    to false)
                "replace": False # Replaces the old file if already exists
                    (optional,  defaults to false)
            }
        
    Returns:
        location: string: The location of the file in the cloud, None if failed
    """
    import tempfile
    
    # Validate inputs
    if filePath is None and obj is None:
        raise ValueError("Either filePath or obj must be provided")
    if obj is not None and objName is None:
        raise ValueError("objName must be provided when uploading an object")
    
    # Asserts for info dictionary
    assert info is not None, "info dictionary is required"
    assert "platform" in info.keys(), "Platform not specified in info dictionary"
    assert "dirName" in info.keys(), "dirName not specified in info dictionary"
    assert "api" in info.keys(), "Enter an api object to interact with"
    assert info["platform"] == "huggingface", "Only huggingface platform is supported"
    assert "repoID" in info.keys(), "repoID not specified in info dictionary" # f"{your_username}/{your_dataset_name}"
    
    temp_file_path = None
    try:
        # Handle object saving using saveModel function
        if obj is not None:
            # Create temporary file path for the object
            # Check if objName already has .pth extension to avoid double extension
            if objName.endswith('.pth'):
                temp_file_path = f"{tempfile.gettempdir()}/{objName}"
            else:
                temp_file_path = f"{tempfile.gettempdir()}/{objName}.pth"
            
            # Use the existing saveModel function to save the object
            saveModel(obj, temp_file_path, backup=False)
            
            # Use the temporary file as the upload source
            upload_file_path = temp_file_path
        else:
            # Use the provided file path
            assert os.path.isfile(filePath), "File not found"
            upload_file_path = filePath
        
        # Setup the necessary variables
        __api = info["api"]
        __repoID = info["repoID"]
        __dirName = info["dirName"]
        
        # Create the file path in repo with directory structure
        if __dirName and __dirName.strip():  # Check if dirName is not empty
            __filePathInRepo = f"{__dirName}/{os.path.basename(upload_file_path)}"
        else:
            __filePathInRepo = f"{os.path.basename(upload_file_path)}"
        
        # Create the repo if it doesn't exist
        try:
            __api.create_repo(repo_id = __repoID, repo_type = "model", private = False if "private" not in info.keys() else info["private"])
        except Exception as repo_error:
            # Repository might already exist, continue with upload
            pass
        
        # Check if file exists in the repository and delete it
        if (info["replace"] if "replace" in info.keys() else False):
            try:                
                # Try to delete the existing file
                __api.delete_file(
                    path_in_repo = __filePathInRepo,
                    repo_id = __repoID
                )
            except Exception as delete_error:
                # File might not exist, continue with upload
                pass
        
        # Push the dataset to the Hugging Face Hub
        msg = __api.upload_file(
            path_or_fileobj = upload_file_path,
            path_in_repo = __filePathInRepo, 
            repo_id = __repoID
        )
        
        return msg
        
    except Exception as e:
        print(e)
        return None
    finally:
        # Clean up temporary file if created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass  # Ignore cleanup errors

def loadNetwork(fileName, algorithm, **kwargs):
    """
    Loads the previous training details from a file. The file should be 
    a dictionary, with all the details necessary to pickup where we left.
    The necessary data are explained below:
    
    Args:
        fileName (str): The name of the file
        algorithm (str): The type of the algorithm, either "DDQN" or "PPO"
        kwargs (dict): A dictionary with the following keys:
            algorithm: "DDQN":
            qNetwork_model (torch.nn): The qNetwork_model
            optimizer_main (torch.optim): The qNetwork_model's optimizer 
                object
            targetQNetwork_model (torch.nn): The targetQNetwork_model
            trainingParams (list): A list containing following parameters
                in order. We chose this approach to be able to change training 
                parameters in-place:
                startEpisode (int): The episode number to start from
                startEbsilon (int): The starting ebsilon number (The ebsilon 
                    prior to latest run's termination)
                lstHistory (list): The list holding the training history
                eDecay (float): The decay of ebsilon
                NUM_ENVS (int): The number of agents
                mem (ReplayMemory): An instance of replay memory object
            
            algorithm: "PPO":
            actor_network (torch.nn): The actor_network
            critic_network (torch.nn): The critic_network
            optimizer_actor (torch.optim): The actor network's optimizer 
            optimizer_critic (torch.optim): The critic network's optimizer 
            trainingParams (list): A list containing following parameters
                in order. We chose this approach to be able to change training 
                parameters in-place:
                startEpisode (int): The episode number to start from
                lstHistory (list): The list holding the training history
    """
    # Check if all necessary data has been given so it can be overwritten 
    # when loaded and passed back to the user
    if algorithm == "DDQN":
        assert "qNetwork_model" in kwargs.keys(), "Please pass the qNetwork_model object"
        assert "optimizer_main" in kwargs.keys(), "Please pass the optimizer_main object"
        assert "targetQNetwork_model" in kwargs.keys(), "Please pass the targetQNetwork_model object"
        assert "trainingParams" in kwargs.keys(), "Please pass the trainingParams object"
        if len(kwargs["trainingParams"]) != 6: print( f"You should enter the following parameters in the order: startEpisode, startEbsilon, lstHistory, eDecay, mem.")
        
        if os.path.isfile(fileName):
            try:
                # Try to read the main file
                try:
                    __data = torch.load(fileName, weights_only = False)
                except:
                    print("Couldn't load the main file, trying to load the backup file")
                    try:
                        # Try to read the backup file
                        __data = torch.load(os.path.join(os.path.dirname(fileName), "Backups", os.path.basename(fileName)), weights_only = False)
                    except Exception as e:
                        raise Exception(f"Couldn't load the backup file")
                
                # Check if the file is a dictionary
                if not isinstance(__data, dict): raise Exception(f"Couldn't load the file. File {fileName} is not a dictionary")

                # Load Q-Network
                kwargs["qNetwork_model"].load_state_dict(__data["qNetwork_state_dict"]) # Model weights
                kwargs["optimizer_main"].load_state_dict(__data["qNetwork_optimizer_state_dict"]) # Optimizer

                # Load target Q-Network
                kwargs["targetQNetwork_model"].load_state_dict(__data["targetQNetwork_state_dict"]) # Model weights

                # Load process parameters
                kwargs["trainingParams"][0] = __data["episode"] # Starting episode number
                kwargs["trainingParams"][1] = __data["hyperparameters"]["ebsilon"] # Starting ebsilon
                kwargs["trainingParams"][2] = __data["train_history"]
                kwargs["trainingParams"][3] = __data["hyperparameters"]["eDecay"]
                kwargs["trainingParams"][4] = __data["hyperparameters"]["NUM_ENVS"]

                kwargs["trainingParams"][5].loadExperiences(
                    __data["experiences"]["state"],
                    __data["experiences"]["action"],
                    __data["experiences"]["reward"],
                    __data["experiences"]["nextState"],
                    __data["experiences"]["done"],
                )
                
                # All changes are in-place, however, we return the changed objects for convenience
                return (
                    kwargs["qNetwork_model"],
                    kwargs["optimizer_main"],
                    kwargs["targetQNetwork_model"],
                    kwargs["trainingParams"][0],  # startEpisode
                    kwargs["trainingParams"][1],  # startEbsilon
                    kwargs["trainingParams"][2],  # lstHistory
                    kwargs["trainingParams"][3],  # eDecay
                    kwargs["trainingParams"][4],  # environment/agent number
                    kwargs["trainingParams"][5],  # mem
                )

            except Exception as e:
                print("ERROR: ", e)
                return None
        else:
            raise Exception(f"Couldn't load the file. File {fileName} does not exist")
    elif algorithm == "PPO":
        assert "actor_network" in kwargs.keys(), "Please pass the actor_network object"
        assert "critic_network" in kwargs.keys(), "Please pass the critic_network object"
        assert "optimizer_actor" in kwargs.keys(), "Please pass the optimizer_actor object"
        assert "optimizer_critic" in kwargs.keys(), "Please pass the optimizer_critic object"
        assert "trainingParams" in kwargs.keys(), "Please pass the trainingParams object"
        if len(kwargs["trainingParams"]) != 2: print( f"You should enter the following parameters in the order: startEpisode, lstHistory.")
        
        if os.path.isfile(fileName):
            try:
                # Try to read the main file
                try:
                    __data = torch.load(fileName, weights_only = False)
                except:
                    print("Couldn't load the main file, trying to load the backup file")
                    try:
                        # Try to read the backup file
                        __data = torch.load(os.path.join(os.path.dirname(fileName), "Backups", os.path.basename(fileName)), weights_only = False)
                    except Exception as e:
                        raise Exception(f"Couldn't load the backup file")
                
                # Check if the file is a dictionary
                if not isinstance(__data, dict): raise Exception(f"Couldn't load the file. File {fileName} is not a dictionary")
                
                # Actor network
                kwargs["actor_network"].load_state_dict(__data["actor_network_state_dict"]) # Model weights
                kwargs["optimizer_actor"].load_state_dict(__data["optimizer_actor_state_dict"]) # Optimizer

                # Critic network
                kwargs["critic_network"].load_state_dict(__data["critic_network_state_dict"]) # Model weights
                kwargs["optimizer_critic"].load_state_dict(__data["optimizer_critic_state_dict"]) # Optimizer

                # Load process parameters
                kwargs["trainingParams"][0] = __data["episode"] # Starting episode number
                kwargs["trainingParams"][1] = __data["train_history"]
                
                # All changes are in-place, however, we return the changed objects for convenience
                return (
                    kwargs["actor_network"],
                    kwargs["critic_network"],
                    kwargs["optimizer_actor"],
                    kwargs["optimizer_critic"],
                    kwargs["trainingParams"][0],  # startEpisode
                    kwargs["trainingParams"][1],  # lstHistory
                    kwargs["trainingParams"][2],  # timestep
                )

            except Exception as e:
                print("ERROR: ", e)
                return None
        else:
            raise Exception(f"Couldn't load the file. File {fileName} does not exist")

def modelParamParser():
    """
    Gets the arguments from the command line for the model to run
    """
    # Make the argument parser
    parser = argparse.ArgumentParser(description = "The neural network with specified parameters")
    
    # Add necessary arguments
    parser.add_argument("--name", "-n", type = str, default = "unknown", help = "The name of the model")
    parser.add_argument("--env", "-e", type = str, default = "unknown", help = "The environment name")
    parser.add_argument("--env_options", "-eops", type = str, default = "{}", help = "The options for the environment in JSON format")
    parser.add_argument("--algorithm", "-alg", type = str, default = "DDQN", help = "The RL algorithm to use (PPO or DDQN)")
    parser.add_argument("--algorithm_options", "-algops", type = str, default = "", help = "The options for the RL algorithm in JSON format")
    parser.add_argument("--network", "-net", type = str, default = "", help = "The network architecture to use (ann or snn)")
    parser.add_argument("--network_options", "-netops", type = str, default = "", help = "The options for the network in JSON format")
    parser.add_argument("--network_actor", "-net_actor", type = str, default = "", help = "The network_actor architecture to use (ann or snn)")
    parser.add_argument("--network_actor_options", "-netops_actor", type = str, default = "", help = "The options for the network_actor in JSON format")
    parser.add_argument("--network_critic", "-net_critic", type = str, default = "", help = "The network_critic architecture to use (ann or snn)")
    parser.add_argument("--network_critic_options", "-netops_critic", type = str, default = "", help = "The options for the network_critic in JSON format")

    parser.add_argument("--continue_run", "-c", action = "store_true", help = "Continue the last run")
    parser.add_argument("--agents", "-a", type = int, default = 1, help = "Number rof agents")
    parser.add_argument("--extra_info", "-extra", type = str, default = "", help = "Extra information")
    parser.add_argument("--max_run_time", "-t", type = int, default = 60 * 60, help = "Maximum run time of training in seconds")
    parser.add_argument("--stop_learning_at_win_percent", "-slw", type = float, default = 0.995, help = "Stop updating the network if the last 100 episodes' win percent is greater than this value")
    parser.add_argument("--upload_to_cloud", "-u", action = "store_true", help = "Upload the training history to cloud")
    parser.add_argument("--local_backup", "-l", action = "store_true", help = "Save the training networks' data locally")
    parser.add_argument("--debug", "-db", action = "store_true", help = "Keep track of the training progress")
    parser.add_argument("--train_finish_timestamp", "-tfts", type = float, default = 0., help = "The timestamp at which entire training (all runs) is finished")
    parser.add_argument("--stop_condition", "-stopcond", type = str, default = "", help = "The training stop conditions")
    parser.add_argument("--finished", "-f", action = "store_true", help = "Is the training finished")
    
    return parser

# For visualizition

def plot_action_distribution(action_counts, ax=None):
    """
    Plot action distribution across episodes as stacked bar chart.
    
    Parameters:
    action_counts: list of lists or 2D array
        Shape: (num_episodes, num_actions)
        Each row contains counts of actions [action0, action1, action2, action3, action4]
        for that episode
    ax: matplotlib axis object (optional)
        If None, creates a new figure and axis
    """
    action_counts = np.array(action_counts)
    num_episodes = action_counts.shape[0]
    num_actions = action_counts.shape[1]
    
    # Episode numbers for x-axis
    episodes = range(1, num_episodes + 1)
    
    # Colors for different actions
    colors = ['red', 'green', 'black', 'blue', 'orange']
    
    # Create the plot if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    # Create stacked bar chart
    bottom = np.zeros(num_episodes)
    for i in range(num_actions):
        ax.bar(episodes, action_counts[:, i], bottom=bottom, label=f'Action {i}', color=colors[i % len(colors)], width = 1)
        bottom += action_counts[:, i]
    
    # Customize the plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Number of Actions', fontsize=12)
    ax.set_title('Action Distribution Across Episodes')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    

    
    plt.tight_layout()
    return fig, ax

def get_next_run_number_and_create_folder(continueLastRun = False, args = None):
    """
    Searches for existing folders in runs_data and returns the next available run number and folder path

    Args:
        continueLastRun (bool): If True, continues the last run instead of creating a new one
        args (Namespace): The configuration arguments. To check if the passed configuration 
            for the training is the same as the last run. If not, a new folder will be created 
            even if continueLastRun is True.
    """
    # Define the runs_data directory path
    runs_data_dir = "runs_data"
    
    # Create runs_data directory if it doesn't exist
    if not os.path.exists(runs_data_dir):
        os.makedirs(runs_data_dir)
    
    # Get list of existing folders in runs_data
    existing_folders = [f for f in os.listdir(runs_data_dir) if os.path.isdir(os.path.join(runs_data_dir, f))]

    # Filter for folders with numeric names and find the highest number
    run_numbers = [int(f) for f in existing_folders if f.isdigit()]

    if not continueLastRun or run_numbers == []:
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run_number = max(run_numbers) if run_numbers else 1

        # Check if the configuration has changed
        if args is not None:
            with open(os.path.join(runs_data_dir, str(next_run_number), "conf.json"), 'r') as file:
                pastConfig = json.load(file)
                currentConfig = vars(args)

                def _normalize(v):
                    if isinstance(v, str):
                        s = v.strip()
                        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                            try:
                                return json.loads(s)
                            except Exception:
                                return v
                    return v

                for key in pastConfig.keys():
                    if key in currentConfig:
                        past = pastConfig.get(key)
                        curr = _normalize(currentConfig[key])
                        if past != curr:
                            print(f"Configuration for '{key}' has changed from {past} to {curr}. Creating a new run folder.")
                            next_run_number += 1
                            break
    
    # Create new folder with the next run number
    new_folder_path = os.path.join(runs_data_dir, str(next_run_number))
    os.makedirs(new_folder_path, exist_ok=True)
    
    return next_run_number, new_folder_path

# Utility functions for plotting the training history and progress

def plotEpisodeReward(df, saveLoc):
    # Calculate 100-step moving average
    df['Moving_Average'] = df.points.rolling(window=100).mean()

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.episode, df.points, label='Points', color='blue')
    plt.plot(df.episode, df.Moving_Average, label='100 Episode Average', color='red', linewidth=1)
    plt.title('Data with 100-Step Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(saveLoc)
    plt.close()

def plotTrainingProcess(df, saveLoc):
    # get cumulative sum of termianted or trunctated episodes
    df["isTerminated"] = df["state"] == "terminated"
    df["terminatedSum"] = df["isTerminated"].cumsum()
    df["isTruncated"] = df["state"] == "truncated"
    df["truncatedSum"] = df["isTruncated"].cumsum()

    df["wonEpisode"] = 75 < df["finalEpisodeReward"]
    df["wonEpisodeCount"] = df["wonEpisode"].cumsum()
    df["wonEpisodeCountLas100"] = df["wonEpisode"].rolling(window=100).sum()
    df["wonEpisodeCountPercent"] = df["wonEpisodeCount"] / (df["episode"] + 1) * 100
    df["wonEpisodeCountLas100Percent"] = df["wonEpisodeCountLas100"] / 100 * 100

    actionCounts = np.array(df['nActionInEpisode'].tolist())

    # Process the layer-wise spikes safely
    layerWiseSpikes = None
    if 'spikesPerLayer' in df.columns:
        _lst_raw = df['spikesPerLayer'].tolist()
        # keep only iterable per-episode spike sequences (filter out None)
        _lst = [x for x in _lst_raw if isinstance(x, (list, tuple, np.ndarray))]
        if len(_lst) > 0:
            try:
                layerWiseSpikes = [list(tup) for tup in zip(*_lst)]
            except TypeError:
                # In case any unexpected non-iterable slips in
                layerWiseSpikes = None

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(8, 15))

    # Plot
    ax1.set_title("Number of terminated and truncated episodes")
    ax1.plot(df.episode, df.terminatedSum, label='terminated', color='blue')
    ax1.plot(df.episode, df.truncatedSum, label='truncated', color='red')
    ax1.legend()

    # Plot
    ax2.set_title("Percentage of won episodes")
    ax2.plot(df.episode, df.wonEpisodeCountPercent, label='All episodes', color='blue')
    ax2.plot(df.episode, df.wonEpisodeCountLas100Percent, label='Last 100 episodes', color='red')
    ax2.set_ylim(0, 100)
    ax2.legend()

    # Plot
    ax3.set_title("Average Total Spikes")
    ax3.scatter(df.episode, df.avgSpikes, label='AVG spike number', color='blue', s = 3)
    ax3.legend()

    # Plot
    if layerWiseSpikes is not None:
        ax4.set_title("Layer-wise Spikes")
        ax4.set_yscale('log')
        for i in range(len(layerWiseSpikes)):
            ax4.scatter(df.episode, layerWiseSpikes[i], label=f'layer {i}' if i != len(layerWiseSpikes) - 1 else 'Output', s = 3)
        ax4.legend()
    else:
        # Hide the axis if no valid data is available
        ax4.set_visible(False)

    # Plot
    plot_action_distribution(actionCounts, ax5)
    ax5.set_title("Action distribution")
    ax5.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plot
    plt.savefig(saveLoc)
    plt.close()

def plotGradientNorms(df, saveLoc):
    totalNormHist = df['totalGradientNorms'].tolist()
    
    dfSeparated = pd.json_normalize(df['layerWiseNorms']).dropna()
    names = dfSeparated.columns
    nPlots = int(dfSeparated.shape[1] / 2) + 1

    fig, axes = plt.subplots(nPlots, 1, sharex=True, figsize=(8, 3 * nPlots))

    for i in range(nPlots - 1):
        ax = axes[i]
        ax.set_yscale('log')
        ax.scatter(range(len(dfSeparated.iloc[:, i*2])), dfSeparated.iloc[:, i*2], label=f'{names[i*2].replace(".", " ")}', s = 3)
        ax.scatter(range(len(dfSeparated.iloc[:, i*2 + 1])), dfSeparated.iloc[:, i*2 + 1], label=f'{names[i*2 + 1].replace(".", " ")}', s = 3)
        ax.legend()

    axes[-1].scatter(range(len(totalNormHist)), totalNormHist, label='Total Norm', s = 3)
    axes[-1].legend()
    axes[-1].set_yscale('log')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plot
    plt.savefig(saveLoc)
    plt.close()
