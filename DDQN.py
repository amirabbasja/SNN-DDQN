import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from utils import *
from models import *


class DDQN():
    def __init__(self, sessionName, args, networks):
        """
        Initialize the DDQN agent with the given arguments.

        Args:
            sessionName (str): The name of the session, mostly the server's given name in which we are training
            
            args (dict): A dictionary containing hyperparameters and configurations. Must include:
                - General parameters:
                - name: Project name (Different from session name)
                - hidden_layers: List of integers defining the number of nodes in each hidden layer
                - learning_rate: Learning rate for the optimizer
                - extra_info: Additional information as a string
                - continue_run: Boolean flag to continue from last checkpoint
                - debug: Boolean flag for debug mode, in which, extra data is saved in a directory for debugging purposes
                - stop_learning_at_win_percent: Percentage threshold of latest 100 won episodes where we stop updating networks
                - upload_to_cloud: Boolean flag to upload the model and training data to the cloud
                - local_backup: Boolean flag to save a local backup of the model and training data
                - agents: Number of parallel agents/environments to run (Defaults to 1)
                - nEpisodes: Total number of episodes to train
                - maxNumTimeSteps: Maximum number of time steps per episode
                - max_run_time: Maximum run time for this training session (in seconds)
                - run_save_path: The path to save the training data and model
                - action_space: List of available actions to take in the environment
                - env: The environment object
                - stateSize: The size of the state space
                
                - DDQN specific parameters:
                - batch: Mini-batch size for training 
                - gamma: Discount factor for future rewards
                - memorySize: Size of the replay memory
                - startEbsilon: Starting value of epsilon for exploration
                - decay: Epsilon decay rate for exploration
                - endEbsilon: Minimum value of epsilon for exploration
                - numUpdateTS: The timestep frequency at which we need to update the networks

            networks (dict): A dictionary containing the neural network object's to be used in training. Must include:
                - architecture: Network architecture type (e.g., 'ann' or 'snn')
                - qNetwork_model: The main Q-network model (PyTorch nn.Module)
                - targetNetwork_model: The target Q-network model (PyTorch nn.Module)
                - optimizer_main: The optimizer for the main Q-network (PyTorch optimizer)
                - optimizer_target: The optimizer for the target Q-network (PyTorch optimizer)
        """

        # Check if all required arguments are present
        # args - General parameters for training
        assert "name" in args, "Project name is required in args"
        assert "extra_info" in args, "Extra info flag is required in args"
        assert "continue_run" in args, "Continue run flag is required in args"
        assert "debug" in args, "Debug mode flag is required in args"
        assert "stop_learning_at_win_percent" in args, "Stop learning percent is required in args"
        assert "upload_to_cloud" in args, "Upload to cloud flag is required in args"
        assert "local_backup" in args, "Local backup flag is required in args"
        assert "max_run_time" in args, "Maximum run time for this training session is required in args"
        assert "run_save_path" in args, "The path to save the training data and model is required in args"
        assert "action_space" in args, "Action space is required in args (Should be a list of available actions)"
        assert "env" in args, "The environment object is required in args"
        assert "stateSize" in args, "The size of the state space is required in args"
        assert "stop_condition" in args, "Stop conditions should be emphasized in stop_condition"

        # args - Specific parameters for DDQN
        assert "algorithm" in args, "Algorithm is required in args"
        assert "algorithm_options" in args, "Algorithm options are required in args"
        assert "learning_rate" in args["algorithm_options"], "Learning rate is required in args"
        assert "decay" in args["algorithm_options"], "Ebsilon's decay rate is required in args"
        assert "batch" in args["algorithm_options"], "Batch size is required in args"
        assert "gamma" in args["algorithm_options"], "Discount factor is required in args"
        assert "memorySize" in args["algorithm_options"], "Memmory size is required for making the replay memory"
        assert "startEbsilon" in args["algorithm_options"], "Starting value of ebsilon is required in args"
        assert "endEbsilon" in args["algorithm_options"], "Ending value of ebsilon is required in args"
        assert "numUpdateTS" in args["algorithm_options"], "The timestep frequency at which we need to update the networks is required in args"
        assert "nEpisodes" in  args["algorithm_options"], "Number of episodes is required in args"
        assert "maxNumTimeSteps" in args["algorithm_options"], "Maximum number of time steps per episode is required in args"

        # networks
        assert "network" in args, "network is required in args (e.g., ann or snn)"
        assert "network_options" in args, "network options are required in args"
        assert "hidden_layers" in args["network_options"], "Hidden layers configuration is required in args"
        if args["network"] == "snn":
            assert "snn_beta" in args["network_options"], "SNN beta parameter is required in args for SNN"
            assert "snn_tSteps" in args["network_options"], "SNN time steps parameter is required in args for SNN"
        assert "qNetwork_model" in networks, "Q-network model is required in networks dictionary"
        assert "targetNetwork_model" in networks, "Target Q-network model is required in networks dictionary"
        assert "optimizer_main" in networks, "Main optimizer is required in networks dictionary"
        assert "optimizer_target" in networks, "Target optimizer is required in networks dictionary"

        if "agents" in args: 
            if 1 < args["agents"]:raise Exception("For now, only 1 agent is supported. Setting agents to 1.") #TODO: Add support for parallel agents

        if args["algorithm"] != "DDQN":
            raise ValueError(f"Algorithm should be DDQN, not {args['algorithm']}")
        
        if args["algorithm_options"] == None or args["algorithm_options"] == {}:
            raise ValueError("algorithm_options cannot be None or empty for DDQN")
        
        if not("maxEpisodes" in args["stop_condition"] or "maxAvgPoint" in args["stop_condition"]):
            raise ValueError("Stop conditions should be emphasized in stop_condition, either maxEpisodes or maxAvgPoint should be emphasized")

        # Set pytorch parameters: The device (CPU or GPU) and data types
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.dtype = torch.float

        # Set training hyperparameters
        self.sessionName = sessionName
        self.projectName = args["name"]
        self.extraInfo = args["extra_info"] if 'extra_info' in args else ""
        self.continueLastRun = args["continue_run"]
        self.debugMode = args["debug"]
        self.stopLearningPercent = args["stop_learning_at_win_percent"]
        self.uploadToCloud = args["upload_to_cloud"]
        self.localBackup = args["local_backup"]
        self.NUM_ENVS = args["agents"] if 'agents' in args else 1
        self.maxRunTime = args["max_run_time"]
        self.uploadInfo = args["uploadInfo"] if self.uploadToCloud else {}
        self.runSavePath = args["run_save_path"]
        self.actionSpace = args["action_space"]
        self.nActions = len(self.actionSpace)
        self.env = args["env"]
        self.stateSize = args["stateSize"]
        self.stop_condition = args["stop_condition"]

        # DDQN hyperparameters
        _options = args["algorithm_options"]
        self.learningRate = _options["learning_rate"]#
        self.eDecay = _options["decay"]#
        self.gamma = _options["gamma"]#
        self.miniBatchSize = _options["batch"]#
        self.memorySize = _options["memorySize"]#
        self.agentExp = namedtuple("exp", ["state", "action", "reward", "nextState", "done"])
        self.nEpisodes = _options["nEpisodes"]
        self.maxNumTimeSteps = _options["maxNumTimeSteps"]
        self.ebsilon = _options["startEbsilon"]#
        self.endEbsilon   = _options["endEbsilon"]#
        self.numUpdateTS = _options["numUpdateTS"]#

        self.avgWindow = 100 # The number of previous episodes for calculating the average episode reward
    
        # Handle the online/offline model saving parameters
        self.hiddenNodes = args["network_options"]["hidden_layers"]
        self.backUpData = {}
        self.modelDetails = f"{'_'.join([str(l) for l in self.hiddenNodes])}_{self.learningRate}_{self.eDecay}_{self.miniBatchSize}_{self.gamma}_{self.NUM_ENVS}_{self.extraInfo}"
        if self.uploadToCloud:
            self.uploadInfo["dirName"] = f"./{sessionName}-{self.projectName}_{self.modelDetails}"
        self.saveFileName = f"{self.projectName}_{self.modelDetails}.pth"

        self.mem = ReplayMemory(self.memorySize, self.dtype, self.device)

        # Neural network models and optimizers
        self.networkArchitecture = args["network"]
        self.qNetwork_model = networks['qNetwork_model'].to(self.device, dtype = self.dtype)
        self.targetQNetwork_model = networks['targetNetwork_model'].to(self.device, dtype = self.dtype)
        self.optimizer_main = networks['optimizer_main']
        self.optimizer_target = networks['optimizer_target']

        self.startEpisode = 0
        self.startEbsilon = None
        self.lstHistory = None
        self.avgReward = -float("inf")
        if self.continueLastRun and os.path.isfile(os.path.join(self.runSavePath, self.saveFileName)):
            # Load necessary parameters to resume the training from most recent run 
            load_params = {
                "qNetwork_model": self.qNetwork_model,
                "optimizer_main": self.optimizer_main,
                "targetQNetwork_model": self.targetQNetwork_model,
                "trainingParams": [
                    self.startEpisode, 
                    self.startEbsilon, 
                    self.lstHistory, 
                    self.eDecay, 
                    self.NUM_ENVS, 
                    self.mem
                ]
            }

            # NUM_ENVS is a constant and is defined when running the script for the first time, So we disregard re-loading it
            self.qNetwork_model, self.optimizer_main, self.targetQNetwork_model, self.startEpisode, self.startEbsilon, self.lstHistory, self.eDecay, _, self.mem = loadNetwork(os.path.join(self.runSavePath, self.saveFileName), "DDQN", **load_params)
            
            self.qNetwork_model = self.qNetwork_model.to(self.device, dtype = self.dtype)
            self.targetQNetwork_model = self.targetQNetwork_model.to(self.device, dtype = self.dtype)

            self.ebsilon = self.startEbsilon if self.startEbsilon != None else self.ebsilon
            print("Continuing from episode:", self.startEpisode)

        print(f"Device is: {self.device}")

    def getAction(self, qVal: list, e:float, actionSpace: list, device: torch.device ) -> int:
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

    def updateNetworks(self, timeStep: int, replayMem: ReplayMemory, miniBatchSize: int, C: int) -> bool:
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

    def decayEbsilon(self, currE: float, rate:float, minE:float) -> float:
        """
        Decreases ebsilon each time called. It multiplies current ebsilon to decrease rate.
        The decreasing is continued until reaching minE.
        """
        return(max(currE*rate, minE))

    def computeLoss(self, experiences:tuple, gamma:float, qNetwork, target_qNetwork):
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

    def fitQNetworks(self, experience, gamma, qNetwork, target_qNetwork):
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
        loss = self.computeLoss(experience, gamma, __qNetworkModel, __targetQNetworkModel)

        __qNetworkModel.train()
        __targetQNetworkModel.train()

        __qNetworkOptim.zero_grad()
        loss.backward()
        __qNetworkOptim.step()

        # Update the target Q network's weights using soft updating method
        for targetParams, primaryParams in zip(__targetQNetworkModel.parameters(), __qNetworkModel.parameters()):
            targetParams.data.copy_(targetParams.data * (1 - .001) + primaryParams.data * .001)

    def _printProgress(self, delay, lastPrintTime, trainingStartTime, **kwargs):
        """
        Prints the training progress in a manner that doesn't flood the terminal.
        The print happens in delay second intervals to avoid jitters.

        Args:
            delay (int): The minimum time (in seconds) between two prints (In seconds)
            lastPrintTime (int): The time the last print happened
            trainingStartTime (int): The timestamp when the training started
            **kwargs: Should have following keys
                - episode (int): The current episode number
                - epPointAvg (float): The average of the last {numP_Average} episodes
                - finishTime (int): The finish timestamp of the current run
                - latestCheckpoint (int): The latest checkpoint episode number
                - t (int): The current timestep in the episode
                - episodeStartTime (int): The timestamp when the current episode started
        """

        # Print the training status. Print only once each delay second to avoid jitters.
        if delay < (time.time() - lastPrintTime):
            os.system('cls' if os.name == 'nt' else 'clear')
            lastPrintTime = time.time()
            print(f'ElapsedTime: {int(time.time() - trainingStartTime):<5}s | Episode: {kwargs["episode"]:<5} | Timestep: {kwargs["t"]:<5} | The average of the {self.avgWindow:<5} episodes is: {kwargs["epPointAvg"]:<5.2f}')
            print(f'Latest chekpoint: {kwargs["latestCheckpoint"]} | Speed {kwargs["t"]/(time.time()-kwargs["episodeStartTime"]+1e-9):.1f} tps | ebsilon: {self.ebsilon:.3f}')
            print(f'Remaining time of this run: {kwargs["finishTime"] - time.time():.1f}s')
            print(f"Memory details: {self.mem.len}")
        
        return lastPrintTime

    def _stopTraining_maxEpisodes(self, episode):
        """
        Returns True if maxEpisodes is reached. The 
        """
        if self.stop_condition["maxEpisodes"] <= episode: return True
        else: return False

    def _stopTraining_maxAvgPoint(self, epPointAvg):
        """
        Returns True if maxAvgPoint is reached
        """
        if self.stop_condition["maxAvgPoint"] <= epPointAvg: return True
        else: return False

    def train(self):
        """
        Starts the training process
        """
        __trainingStartTime = time.time()
        __finishTime = __trainingStartTime + self.maxRunTime
        __device = self.device
        __dtype = self.dtype

        print("Starting training...")
        episodePointHist = [] # For saving each episode's point for later demonstration
        episodeHistDf = None
        self.lstHistory = [] if self.lstHistory == None else self.lstHistory
        self.avgReward = -float("inf") if len(self.lstHistory) == 0 else pd.DataFrame(self.lstHistory).iloc[-self.avgWindow:]["points"].mean()
        latestCheckpoint = 0
        _lastPrintTime = 0
        self._last100WinPercentage = 0

        for episode in range(self.startEpisode, self.nEpisodes):
            initialSeed = random.randint(1,1_000_000_000) # The random seed that determines the episode's I.C.
            self.state, self.info = self.env.reset(seed = initialSeed)
            points = 0
            actionString = ""

            tempTime = time.time()

            if self.debugMode:
                _nActionInEpisode = np.array([0 for _ in range(self.nActions)])

                if self.networkArchitecture == "snn":
                    _totalSpikes = 0
                    _spikesPerLayer = [0 for _ in range(len(self.qNetwork_model.layers))]
                
            # Run a single episode
            for t in range(self.maxNumTimeSteps):
                qValueForActions, trainInfo = self.qNetwork_model(torch.tensor(self.state, device = __device, dtype = __dtype).unsqueeze(0))

                # use ebsilon-Greedy algorithm to take the new step
                action = self.getAction(qValueForActions, self.ebsilon, self.actionSpace, self.device).cpu().numpy()[0]

                if self.debugMode:
                    _nActionInEpisode[action] += 1
                    _gradientNorms, _layerWiseNorms = computeGradientNorms(self.qNetwork_model)
                    
                    if self.networkArchitecture == "snn":
                        _totalSpikes += trainInfo["totalSpikes"]
                        _spikesPerLayer = [spikes + newFeedForwardSpikes for spikes, newFeedForwardSpikes in zip(_spikesPerLayer, trainInfo["spikesPerLayer"])]

                # Take a step
                observation, reward, terminated, truncated, info = self.env.step(action)

                # Store the experience of the current step in an experience deque.
                self.mem.addNew(self.agentExp(self.state, action,reward, observation,True if terminated or truncated else False))

                if not self.stopLearningPercent < self._last100WinPercentage:
                    # Check to see if we have to update the networks in the current step
                    update = self.updateNetworks(t, self.mem, self.miniBatchSize, self.numUpdateTS)
                    
                    if update:
                        # Update the NNs
                        experience = self.mem.sample(self.miniBatchSize)

                        # Update the Q-Network and the target Q-Network
                        # Bear in mind that we do not update the target Q-network with direct gradient descent.
                        # so there is no optimizer needed for it
                        self.fitQNetworks(experience, self.gamma, [self.qNetwork_model, self.optimizer_main], [self.targetQNetwork_model, None])

                # Save the necessary data
                points += reward
                self.state = observation.copy()
                actionString += f"{action},"

                # Print the training status. Print only once each second to avoid jitters.
                _lastPrintTime = self._printProgress(
                    1, 
                    _lastPrintTime, 
                    __trainingStartTime, 
                    episode = episode, 
                    epPointAvg = self.avgReward, 
                    finishTime = __finishTime, 
                    latestCheckpoint = latestCheckpoint, 
                    t = t, 
                    episodeStartTime = tempTime
                )
                
                if terminated or truncated or __finishTime < time.time(): break
            
            self._last100WinPercentage = np.sum([1 if exp["finalEpisodeReward"] > 75 else 0 for exp in self.lstHistory[-100:]]) / 100

            # Save the episode history
            self.lstHistory.append({
                "episode": episode,
                "seed": initialSeed,
                "points": points,
                "timesteps": t,
                "duration": time.time() - tempTime,
                "finalEpisodeReward": reward, # For deducting if termination is for winning or losing
                "state": "terminated" if terminated else "truncated" if truncated else "none", # terminated or truncated or none
                "totalSpikes": _totalSpikes if self.networkArchitecture == "snn" and self.debugMode else None,
                "avgSpikes": _totalSpikes/t if self.networkArchitecture == "snn" and self.debugMode else None,
                "spikesPerLayer": _spikesPerLayer if self.networkArchitecture == "snn" and self.debugMode else None,
                "avgSpikesPerLayer": [spikes/t for spikes in _spikesPerLayer] if self.networkArchitecture == "snn" and self.debugMode else None,
                "nActionInEpisode": _nActionInEpisode,
                "totalGradientNorms": _gradientNorms if self.debugMode else None,
                "layerWiseNorms": _layerWiseNorms if self.debugMode else None
            })
        
            # Saving the current episode's points and time
            episodePointHist.append(points)

            # Getting the average of {numP_Average} episodes
            self.avgReward = np.mean(episodePointHist[-self.avgWindow:])

            # Decay ebsilon
            self.ebsilon = self.decayEbsilon(self.ebsilon, self.eDecay, self.endEbsilon)

            # Save model weights and parameters periodically (For later use)
            if self.localBackup:
                if (episode + 1) % 100 == 0 or episode == 2:
                    _exp = self.mem.exportExperience()
                    backUpData = {
                        "episode": episode,
                        'qNetwork_state_dict': self.qNetwork_model.state_dict(),
                        'qNetwork_optimizer_state_dict': self.optimizer_main.state_dict(),
                        'targetQNetwork_state_dict': self.targetQNetwork_model.state_dict(),
                        'targetQNetwork_optimizer_state_dict': self.optimizer_target.state_dict(),
                        'hyperparameters': {"ebsilon": self.ebsilon, "eDecay": self.eDecay, "NUM_ENVS": self.NUM_ENVS},
                        "elapsedTime": int(time.time() - __trainingStartTime),
                        "train_history": self.lstHistory,
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
                    saveModel(backUpData, os.path.join(self.runSavePath, self.saveFileName))
            
            # Upload the lightweight progress data to cloud
            if self.uploadToCloud:
                # Only save the history and with lower frequency. 
                # Re-upload no sooner than every n minute (To avoid being rate-limited)
                if lastUploadTime + 120 < time.time():
                    __data = {"train_history": self.lstHistory, "elapsedTime": int(time.time() - __trainingStartTime)}
                    backUpToCloud(obj = __data, objName = f"{self.sessionName}-{self.saveFileName}", info = self.uploadInfo)
                    lastUploadTime = time.time()

            # Save progress info (With higher frequency)
            if (episode + 1) % 5 == 0 or episode == 2:
                # Save the details
                episodeData = {
                    "session_name": self.sessionName,
                    "episode": episode,
                    "reward": points,
                    "avg_reward": self.avgReward,
                    "Win Percentage (last 100)": self._last100WinPercentage,
                }
                
                with open(os.path.join(self.runSavePath, f"training_details.json"), 'w') as f:
                    json.dump(episodeData, f, indent=2)
                

            # Plot the progress
            if (episode + 1) % 100 == 0 or episode == 2:
                # Plot the details
                histDf = pd.DataFrame(self.lstHistory)

                plotEpisodeReward(histDf, os.path.join(self.runSavePath, f"episode_rewards.png"))

                plotTrainingProcess(histDf, os.path.join(self.runSavePath, f"training_process.png"))

                try:
                    plotGradientNorms(histDf, os.path.join(self.runSavePath, f"gradient_norms.png"))
                except Exception as e:
                    print("Could not plot the gradient norms. Error:", e)
            
            # Check stop conditions
            if self._stopTraining_maxAvgPoint(self.avgReward) or self._stopTraining_maxEpisodes(episode): 
                # Change the conf.json and  training_details.json files
                # training_details.json file
                with open(os.path.join(self.runSavePath, f"training_details.json"), 'w') as f:
                    cond1 = self._stopTraining_maxAvgPoint(self.avgReward)
                    cond2 = self._stopTraining_maxEpisodes(episode)
                    stopReason = "maxAvgPoint" if cond1 and not cond2 else "maxEpisodes" if cond2 and not cond1 else  "maxAvgPoint and maxEpisodes" if cond1 and cond2 else None
                    episodeData.update({"stopReason": stopReason})
                    json.dump(episodeData, f, indent=2)

                # conf.json file
                with open(os.path.join(self.runSavePath, f"conf.json"), 'r+') as f:
                    _conf = json.load(f)
                    _conf["finished"] = True
                    f.seek(0)  # Move pointer back to start
                    f.truncate()  # Clear the file
                    json.dump(_conf, f, indent=4)
                
                # conf.json file
                with open('conf.json', 'r+') as f:
                    _conf = json.load(f)
                    _conf["finished"] = True
                    f.seek(0)  # Move pointer back to start
                    f.truncate()  # Clear the file
                    json.dump(_conf, f, indent=4)

                return True
