import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import *
import time, os, random
from utils import plotEpisodeReward, plotTrainingProcess, plotGradientNorms

class PPO:
    def __init__(self, sessionName, args, networks):
        """
        Initializes the PPO agent with the given environment.

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
                - maxNumTimeSteps: Maximum number of time steps per episode
                - max_run_time: Maximum run time for this training session (in seconds)
                - run_save_path: The path to save the training data and model
                - action_space: List of available actions to take in the environment
                - env: The environment object
                - stateSize: The size of the state space
                
                - PPO specific parameters:
                - timeStepsPerBatch: Number of time steps to collect per batch
                - gamma: Discount factor for future rewards
                - nUpdatesPerIteration: Number of update iterations per batch
                - clip: Clipping parameter for PPO

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

        # args - Specific parameters for PPO
        assert "timeStepsPerBatch" in args["algorithm_options"], "Number of time steps per batch is required in args"
        assert "learning_rate" in args["algorithm_options"], "Learning rate is required in args"
        assert "gamma" in args["algorithm_options"], "Discount factor (gamma) is required in args"
        assert "nUpdatesPerIteration" in args["algorithm_options"], "Number of updates per iteration is required in args"
        assert "clip" in args["algorithm_options"], "Clipping parameter is required in args"
        assert "maxNumTimeSteps" in args["algorithm_options"], "Maximum number of time steps per episode is required in args"

        # networks
        assert "network" in args, "network is required in args (e.g., ann or snn)"
        assert "network_options" in args, "network options are required in args"
        assert "hidden_layers_actor" in args["network_options"], "Hidden layers for actor network is required in args"
        assert "hidden_layers_critic" in args["network_options"], "Hidden layers for critic network is required in args"
        if args["network"] == "snn":
            assert "snn_beta" in args["network_options"], "SNN beta parameter is required in args for SNN"
            assert "snn_tSteps" in args["network_options"], "SNN time steps parameter is required in args for SNN"
        assert "actorNetwork" in networks, "actorNetwork model is required in networks dictionary"
        assert "criticNetwork" in networks, "criticNetwork model is required in networks dictionary"
        assert "optimActor" in networks, "Actor network's optimizer is required in networks dictionary"
        assert "optimCritic" in networks, "Critic network's optimizer is required in networks dictionary"

        if "agents" in args: 
            if 1 < args["agents"]: raise Exception("For now, only 1 agent is supported. Setting agents to 1.") #TODO: Add support for parallel agents

        # Set pytorch parameters: The device (CPU or GPU) and data types
        self.device = torch.device("cpu") # Force CPU for now. TODO: Add support for GPU => torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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

        # PPO parameters
        self.learningRate = args["algorithm_options"]["learning_rate"]
        self.timeStepsPerBatch = args["algorithm_options"]["timeStepsPerBatch"]
        self.gamma = args["algorithm_options"]["gamma"]
        self.nUpdatesPerIteration = args["algorithm_options"]["nUpdatesPerIteration"]
        self.clip = args["algorithm_options"]["clip"]
        self.maxNumTimeSteps = args["algorithm_options"]["maxNumTimeSteps"]

        self.avgWindow = 100 # The number of previous episodes for calculating the average episode reward
    
        # Handle the online/offline model saving parameters
        self.backUpData = {}
        self.modelDetails = f"{self.learningRate}_{self.clip}_{self.nUpdatesPerIteration}_{self.gamma}_{self.NUM_ENVS}_{self.extraInfo}"
        if self.uploadToCloud:
            self.uploadInfo["dirName"] = f"./{sessionName}-{self.projectName}_{self.modelDetails}"
        self.saveFileName = f"{self.projectName}_{self.modelDetails}.pth"

        # Neural network models and optimizers
        self.networkArchitecture = args["network"]
        self.actorNetwork = networks['actorNetwork'].to(self.device, dtype = self.dtype)
        self.criticNetwork = networks['criticNetwork'].to(self.device, dtype = self.dtype)
        self.optimActor = networks['optimActor']
        self.optimCritic = networks['optimCritic']
        
        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.nActions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Necessary data to be collected during training
        self._last100WinPercentage = 0.0
        self.lstHistory = None # A list of dictionaries for storing episode history

    def getActions(self, obs):
        """
        Note: actions will be deterministic when testing, meaning that the "mean" 
        action will be our actual action during testing. However, during training 
        we need an exploratory factor, which this distribution can help us with.
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        # # For continuous action spaces, choose below:
        # mean, _ = self.actorNetwork(obs)  # Unpack the tuple to get only the tensor
        # dist = MultivariateNormal(mean, self.cov_mat)
        # action = dist.sample()
        logits, networkInfo = self.actorNetwork(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()  
        logProb = dist.log_prob(action)
        
        # Works only for discrete action spaces
        return int(action.item()), float(logProb.item()), networkInfo
    
    def rollout(self, episodeNumber):
        """
        Collect the data for each batch. Collected data are as follows:

        1. observations: (number of timesteps per batch, dimension of observation)
        2. actions: (number of timesteps per batch, dimension of action)
        3. log probabilities: (number of timesteps per batch)
        4. rewards: (number of episodes, number of timesteps per episode)
        5. reward-to-go's: (number of timesteps per batch)
        6. batch lengths: (number of episodes)

        Args:
            episodeNumber (int): The current episode number. Each rollot can 
            run multiple episodes until the timeStepsPerBatch is reached.
        
        """
        # The data collector
        batchObs = []
        batchActions = []
        batchLogProbs = []
        cumEpisodeRewards = []
        batchRewards = []
        batchRewardsToGo = []
        batchEpisodeLengths = []
        
        t = 0
        while t < self.timeStepsPerBatch:
            # Rewards per episode
            episodeRewards = []
            
            randomSeed = random.randint(0, 1000)
            startTimestep = t
            startTimestamp = time.time()
            obs, info = self.env.reset(seed = randomSeed)
            
            # Debug info
            if self.debugMode:
                _nActionInEpisode = np.array([0 for _ in range(self.nActions)])

                if self.networkArchitecture == "snn":
                    _totalSpikes = 0
                    _spikesPerLayer = [0 for _ in range(len(self.criticNetwork.layers))]

            terminated, truncated = False, False
            for tEpisode in range(self.maxNumTimeSteps):
                t += 1
                
                # Collect observations
                batchObs.append(obs)
                
                action, logProb, networkInfo = self.getActions(obs)
                
                if self.debugMode:
                    _nActionInEpisode[action] += 1
                    _gradientNorms, _layerWiseNorms = computeGradientNorms(self.criticNetwork)
                    
                    if self.networkArchitecture == "snn":
                        _totalSpikes += networkInfo["totalSpikes"]
                        _spikesPerLayer = [spikes + newFeedForwardSpikes for spikes, newFeedForwardSpikes in zip(_spikesPerLayer, networkInfo["spikesPerLayer"])]
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episodeRewards.append(reward)
                # Ensure we append a plain Python int
                batchActions.append(int(action))
                batchLogProbs.append(logProb)
                
                if terminated or truncated: break
            
            episodeNumber += 1

            self._last100WinPercentage = np.sum([1 if exp["finalEpisodeReward"] > 75 else 0 for exp in self.lstHistory[-100:]]) / 100

            # Save the episode history
            self.lstHistory.append({
                "episode": episodeNumber,
                "seed": randomSeed,
                "points": sum(episodeRewards),
                "timesteps": t - startTimestep,
                "duration": time.time() - startTimestamp,
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

            # Plot the progress
            if (episodeNumber + 1) % 100 == 0 or episodeNumber == 2:
                self.plotProgress()

            batchEpisodeLengths.append(tEpisode + 1)
            batchRewards.append(episodeRewards)
            cumEpisodeRewards.append(np.array(episodeRewards).sum())
        
        batchObs = torch.tensor(batchObs, dtype=torch.float32)
        # Use torch.long for Categorical distribution actions | torch.float32 for MultivariateNormal distribution actions
        # Robustly convert actions to a 1D int64 tensor to avoid 'len() of unsized object'
        actions_array = np.asarray(batchActions)
        if actions_array.ndim == 0:
            actions_array = actions_array[None]
        batchActions = torch.from_numpy(actions_array.astype(np.int64))
        batchLogProbs = torch.tensor(batchLogProbs, dtype=torch.float32)
        batchRewardsToGo = self.computeRewardsToGo(batchRewards)
        
        return batchObs, batchActions, batchLogProbs, batchRewardsToGo, batchEpisodeLengths, cumEpisodeRewards
    
    def computeRewardsToGo(self, batchRewards):
        """
        Calculates cumulative future rewards for each time step.
        In each timestep, calculates the total discounted sum of future rewards 
        until the episode ends.
        """
        # The rewards-to-go per episode in each batch
        batchRewardsToGo = []
        
        for episodeRewards in reversed(batchRewards):
            discountedReward = 0
            
            for rew in reversed(episodeRewards):
                discountedReward = rew + discountedReward * self.gamma
                batchRewardsToGo.insert(0, discountedReward)
        
        batchRewardsToGo = torch.tensor(batchRewardsToGo, dtype=torch.float32)
        
        return batchRewardsToGo
    
    def learn(self, totalSteps):
        rewardsMem = []
        criticLossMem = []
        actorLossMem = []
        self.lstHistory = [] if self.lstHistory == None else self.lstHistory

        t = 0
        episode = 0
        _lastPrintTime = time.time()
        _trainingStartTime = time.time()
        _latestCheckpoint = 0
        while t < totalSteps:
            # Collect data
            episodeStartTime = time.time()
            batchObs, batchActions, batchLogProbs, batchRewardsToGo, batchEpisodeLengths, batchRewards = self.rollout(episode)
            episode += len(batchEpisodeLengths)
            rewardsMem.extend(batchRewards)
            
            # Increment time step
            t += np.sum(batchEpisodeLengths)
            
            # Calculate V_{phi,k}
            V, _ = self.evaluate(batchObs, batchActions)

            # Calculate A_{phi,k}
            advantage = self.calculateAdvantage(methods='monte_carlo', batchRewardsToGo = batchRewardsToGo, V = V)
            
            # Normalize the advantage
            advantage = self.noramlizeAdvantage(advantage)
            
            if self._last100WinPercentage < self.stopLearningPercent:
                # Aggregate losses for this batch (to map them to episodes)
                batchActorLosses = []
                batchCriticLosses = []
                for i in range(self.nUpdatesPerIteration):
                    V, currentLogProbs = self.evaluate(batchObs, batchActions)
                    
                    ratio = torch.exp(currentLogProbs - batchLogProbs)
                    
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
                    
                    # Calculate the network losses
                    actorLoss = -torch.min(surr1, surr2).mean()
                    criticLoss = nn.MSELoss()(V, batchRewardsToGo)
                    actorLossMem.append(actorLoss.item())
                    criticLossMem.append(criticLoss.item())

                    # Per-batch aggregation, keeping them to add their average amount to self.lstHistory
                    batchActorLosses.append(actorLoss.item())
                    batchCriticLosses.append(criticLoss.item())
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.optimActor.zero_grad()
                    actorLoss.backward()
                    self.optimActor.step()
                    
                    # Calculate gradients and perform backward propagation for critic network
                    self.optimCritic.zero_grad()
                    criticLoss.backward()
                    self.optimCritic.step()
                
                # Map the average losses of this batch to its episodes
                avgActorLossBatch = float(np.mean(batchActorLosses)) if len(batchActorLosses) > 0 else None
                avgCriticLossBatch = float(np.mean(batchCriticLosses)) if len(batchCriticLosses) > 0 else None
                nEpisodesInBatch = len(batchEpisodeLengths)
                for hist in self.lstHistory[-nEpisodesInBatch:]:
                    hist["avgActorLoss"] = avgActorLossBatch
                    hist["avgCriticLoss"] = avgCriticLossBatch
            else:
                print(f"Skipping the network update as the last 100 episodes win percentage is {self._last100WinPercentage:.2f}% which is above the threshold of {self.stopLearningPercent:.2f}%")
                # Still set None to make plotting easier
                nEpisodesInBatch = len(batchEpisodeLengths)
                for hist in self.lstHistory[-nEpisodesInBatch:]:
                    hist["avgActorLoss"] = None
                    hist["avgCriticLoss"] = None

            # Print the training status. Print only once each 
            _lastPrintTime = self._printProgress(
                1, _lastPrintTime, _trainingStartTime,
                t = t, episode = episode, finishTime = _trainingStartTime + self.maxRunTime, latestCheckpoint = _latestCheckpoint,
                episodeStartTime = episodeStartTime, 
                avgActorLoss = np.array(actorLossMem[-self.avgWindow:-1]).mean(), 
                avgCriticLoss = np.array(criticLossMem[-self.avgWindow:-1]).mean(), 
                avgReward = np.array(rewardsMem[-self.avgWindow:-1]).mean()
            )

        return rewardsMem, actorLossMem, criticLossMem

    def plotProgress(self):
        """
        Plots the training progress including:
            - Episode rewards
            - Training process
        """
        # Plot the progress
        histDf = pd.DataFrame(self.lstHistory)

        plotEpisodeReward(histDf, os.path.join(self.runSavePath, f"episode_rewards.png"))

        plotTrainingProcess(histDf, os.path.join(self.runSavePath, f"training_process.png"))

        self.plotActorCriticLoss(histDf, os.path.join(self.runSavePath, f"actor_critic_loss.png"))

        try:
            plotGradientNorms(histDf, os.path.join(self.runSavePath, f"gradient_norms.png"))
        except Exception as e:
            print("Could not plot the gradient norms. Error:", e)

    def plotActorCriticLoss(self, histDf, savePath):
        """
        Plots the actor and critic loss over episodes.
        Args:
            histDf (pd.DataFrame): DataFrame containing the training history with columns 'episode', 'avgActorLoss', and 'avgCriticLoss'.
            savePath (str): The path to save the generated plot.
        """
        if not {'episode', 'avgActorLoss', 'avgCriticLoss'}.issubset(histDf.columns):
            print("Required columns for plotting actor and critic loss are missing.")
            return

        dfNew = histDf[['episode', 'avgActorLoss', 'avgCriticLoss']]
        dfNew = dfNew.dropna(subset=['avgActorLoss', 'avgCriticLoss'])
        if dfNew.empty:
            print("No valid data for plotting actor and critic loss.")
            return

        fig, (ax_actor, ax_critic) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

        # Actor loss subplot
        ax_actor.plot(dfNew['episode'], dfNew['avgActorLoss'], label='Avg Actor Loss', color='blue')
        ax_actor.set_ylabel('Actor Loss')
        ax_actor.set_title('Actor Loss Over Episodes')
        ax_actor.grid(True)
        ax_actor.legend()

        # Critic loss subplot
        ax_critic.plot(dfNew['episode'], dfNew['avgCriticLoss'], label='Avg Critic Loss', color='red')
        ax_critic.set_xlabel('Episode')
        ax_critic.set_ylabel('Critic Loss')
        ax_critic.set_title('Critic Loss Over Episodes')
        ax_critic.grid(True)
        ax_critic.legend()

        plt.tight_layout()
        plt.savefig(savePath)
        plt.close()

    def _printProgress(self, delay, lastPrintTime, trainingStartTime, **kwargs):
        """
        Prints the training progress in a manner that doesn't flood the terminal.
        The print happens in delay second intervals to avoid jitters.

        Args:
            delay (int): The minimum time (in seconds) between two prints (In seconds)
            lastPrintTime (int): The time the last print happened
            trainingStartTime (int): The timestamp when the training started
            **kwargs: Should have following keys
                - t (int): The current timestep in the episode
                - episode (int): The current episode number
                - finishTime (int): The finish timestamp of the current run
                - latestCheckpoint (int): The latest checkpoint episode number
                - episodeStartTime (int): The timestamp when the current episode started
                - avgActorLoss (float): The average actor loss of the last {numP_Average} episodes
                - avgCriticLoss (float): The average critic loss of the last {numP_Average} episodes
                - avgReward (float): The average reward of the last {numP_Average} episodes
        Returns:
            lastPrintTime (int): The updated time the last print happened
        """
        if delay < (time.time() - lastPrintTime):
            os.system('cls' if os.name == 'nt' else 'clear')
            lastPrintTime = time.time()
            print(f"ElapsedTime: {int(time.time() - trainingStartTime): <5}s | Episode: {kwargs['episode']: <5} | Timestep: {kwargs['t']: <5} | The average of the {self.avgWindow: <5} episodes is: {int(kwargs['avgReward']): <5}")
            print(f"Latest chekpoint: {kwargs['latestCheckpoint']} | Speed {kwargs['t']/(time.time()-kwargs['episodeStartTime']+1e-9):.1f} tps")
            print(f"AVG Actor Loss: {kwargs['avgActorLoss']:.5f} | AVG Critic Loss: {kwargs['avgCriticLoss']:.2f} | AVG EP Reward: {kwargs['avgReward']:.2f}")
            print(f"Remaining time of this run: {kwargs['finishTime'] - time.time():.1f}s")
            
        return lastPrintTime

    def noramlizeAdvantage(self, advantage):
        """
        Noramlizes the advantage
        """
        return (advantage - advantage.mean()) / (advantage.std() + 1e-10)

    def calculateAdvantage(self, methods, **kwargs):
        """
        Calculates the advantage A_{phi,k} based on the selected method.
        Args:
            methods (str): The method to calculate the advantage. Options are:
                - 'monte_carlo': calculates the montecarlo advantage.
                    A_{phi,k} = R_k - V_{phi,k} where  R_k is the reward-to-go. and V_{phi,k} is the value estimated by the critic network.
                - 'gae': calculates the generalized advantage estimation (GAE). #TODO
                - 'td': calculates the temporal difference (TD) advantage. #TODO
            kwargs: additional arguments required for specific methods.
        """
        if methods == 'monte_carlo':
            return self.calculateAdvantage_MC(**kwargs)
        # elif methods == 'gae':
        #     return self.calculateAdvantage_GAE(**kwargs)
        # elif methods == 'td':
        #     return self.calculateAdvantage_TD(**kwargs)
        else:
            raise Exception("The method is not implemented.")

    def evaluate(self, batchObs, batchActs):
        # Query critic network for a value V for each obs in batchObs
        V, _ = self.criticNetwork(batchObs)
        V = torch.squeeze(V)
        
        # # For continuous action spaces, choose below:
        # distMean = self.actorNetwork(batchObs)
        # actionDist = MultivariateNormal(distMean, self.cov_mat)
        # logProbs = actionDist.log_prob(batchActs)

        # For discrete action spaces, choose below:
        logits, _ = self.actorNetwork(batchObs)
        actionDist = Categorical(logits=logits)
        logProbs = actionDist.log_prob(batchActs)
        
        return V, logProbs

    def calculateAdvantage_MC(self, batchRewardsToGo, V):
        """
        Calculates the simplest form of advantage using Monte Carlo estimation.
        A_{phi,k} = R_k - V_{phi,k} where  R_k is the reward-to-go. 
        and V_{phi,k} is the value estimated by the critic network.
        Args:
            batchRewardsToGo (torch.Tensor): The the full discounted return (from time t until episode end).
            baseline (torch.Tensor): The value estimates from the critic network for each time step in the batch.
        Returns:
            advantage (torch.Tensor): The calculated advantage for each time step in the batch.
        """
        # A_{phi,k} = R_k - V_{phi,k}
        return batchRewardsToGo - V.detach()

    def calculateAdvantage_GAE(self):
        # TODO
        pass