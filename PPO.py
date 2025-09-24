import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import time
import os
from models import *

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
        assert "hidden_layers" in args, "Hidden layers configuration is required in args"
        assert "learning_rate" in args, "Learning rate is required in args"
        assert "extra_info" in args, "Extra info flag is required in args"
        assert "continue_run" in args, "Continue run flag is required in args"
        assert "debug" in args, "Debug mode flag is required in args"
        assert "stop_learning_at_win_percent" in args, "Stop learning percent is required in args"
        assert "upload_to_cloud" in args, "Upload to cloud flag is required in args"
        assert "local_backup" in args, "Local backup flag is required in args"
        assert "maxNumTimeSteps" in args, "Maximum number of time steps per episode is required in args"
        assert "max_run_time" in args, "Maximum run time for this training session is required in args"
        assert "run_save_path" in args, "The path to save the training data and model is required in args"
        assert "action_space" in args, "Action space is required in args (Should be a list of available actions)"
        assert "env" in args, "The environment object is required in args"
        assert "stateSize" in args, "The size of the state space is required in args"

        # args - Specific parameters for PPO
        assert "timeStepsPerBatch" in args, "Number of time steps per batch is required in args"
        assert "gamma" in args, "Discount factor (gamma) is required in args"
        assert "nUpdatesPerIteration" in args, "Number of updates per iteration is required in args"
        assert "clip" in args, "Clipping parameter is required in args"

        # networks
        assert "architecture" in args, "Network architecture type is required in args (e.g., ann or snn)"
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
        self.hiddenNodes = args["hidden_layers"]
        self.learningRate = args["learning_rate"]
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
        self.maxNumTimeSteps = args["maxNumTimeSteps"]

        # PPO parameters
        self.timeStepsPerBatch = args["timeStepsPerBatch"]
        self.gamma = args["gamma"]
        self.nUpdatesPerIteration = args["nUpdatesPerIteration"]
        self.clip = args["clip"]
    
        # Handle the online/offline model saving parameters
        self.backUpData = {}
        self.modelDetails = f"{'_'.join([str(l) for l in self.hiddenNodes])}_{self.learningRate}_{self.clip}_{self.nUpdatesPerIteration}_{self.gamma}_{self.NUM_ENVS}_{self.extraInfo}"
        if self.uploadToCloud:
            self.uploadInfo["dirName"] = f"./{sessionName}-{self.projectName}_{self.modelDetails}"
        self.saveFileName = f"{self.projectName}_{self.modelDetails}.pth"

        # Neural network models and optimizers
        self.networkArchitecture = args["architecture"]
        self.actorNetwork = networks['actorNetwork'].to(self.device, dtype = self.dtype)
        self.criticNetwork = networks['criticNetwork'].to(self.device, dtype = self.dtype)
        self.optimActor = networks['optimActor']
        self.optimCritic = networks['optimCritic']
        
        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.nActions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def getActions(self, obs):
        """
        Note: actions will be deterministic when testing, meaning that the "mean" 
        action will be our actual action during testing. However, during training 
        we need an exploratory factor, which this distribution can help us with.
        """
        print(obs)
        obs = torch.tensor(obs, dtype=torch.float32)
        mean = self.actorNetwork(obs)
        print(mean)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        logProb = dist.log_prob(action)
        
        return action.detach().numpy(), logProb.detach()
    
    def rollout(self):
        """
        Collect the data for each batch. Collected data are as follows:

        1. observations: (number of timesteps per batch, dimension of observation)
        2. actions: (number of timesteps per batch, dimension of action)
        3. log probabilities: (number of timesteps per batch)
        4. rewards: (number of episodes, number of timesteps per episode)
        5. reward-to-go's: (number of timesteps per batch)
        6. batch lengths: (number of episodes)
        
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
            
            obs, info = self.env.reset()
            
            terminated, truncated = False, False
            
            for tEpisode in range(self.maxNumTimeSteps):
                t += 1
                
                # Collect observations
                batchObs.append(obs)
                
                action, logProb = self.getActions(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episodeRewards.append(reward)
                batchActions.append(action)
                batchLogProbs.append(logProb)
                
                if terminated or truncated: break
            
            batchEpisodeLengths.append(tEpisode + 1)
            batchRewards.append(episodeRewards)
            cumEpisodeRewards.append(np.array(episodeRewards).sum())
        
        batchObs = torch.tensor(batchObs, dtype=torch.float32)
        batchActions = torch.tensor(batchActions, dtype=torch.float32)
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
        
        t = 0
        while t < totalSteps:
            # Collect data
            batchObs, batchActions, batchLogProbs, batchRewardsToGo, batchEpisodeLengths, batchRewards = self.rollout()
            rewardsMem.extend(batchRewards)
            
            # Increment time step
            t += np.sum(batchEpisodeLengths)
            
            # Calculate V_{phi,k}
            V, _ = self.evaluate(batchObs, batchActions)

            # Calculate A_{phi,k}
            advantage = self.calculateAdvantage(methods='monte_carlo', batchRewardsToGo = batchRewardsToGo, V = V)
            
            # Normalize the advantage
            advantage = self.noramlizeAdvantage(advantage)
            
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
                
                # Calculate gradients and perform backward propagation for actor network
                self.optimActor.zero_grad()
                actorLoss.backward()
                self.optimActor.step()
                
                # Calculate gradients and perform backward propagation for critic network
                self.optimCritic.zero_grad()
                criticLoss.backward()
                self.optimCritic.step()
            print(f"Step: {t} | AVG Actor Loss: {np.array(actorLossMem[-100:-1]).mean():.5f} | AVG Critic Loss: {np.array(criticLossMem[-100:-1]).mean():.2f} | AVG EP Reward: {np.array(rewardsMem[-100:-1]).mean():.2f}")
        
        return rewardsMem, actorLossMem, criticLossMem

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
        V = self.criticNetwork(batchObs).squeeze()
        
        distMean = self.actorNetwork(batchObs)
        actionDist = MultivariateNormal(distMean, self.cov_mat)
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