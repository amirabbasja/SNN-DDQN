from models import *
from huggingface_hub import HfApi, login
from utils import *
import numpy as np
import gymnasium as gym
import torch, os
from PPO import PPO

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
    actorNetwork = qNetwork_ANN([stateSize[0], *args.hidden_layers, nActions])
    criticNetwork = qNetwork_ANN([stateSize[0], *args.hidden_layers, nActions])
elif args.architecture == "snn":
    actorNetwork = qNetwork_SNN([stateSize[0], *args.hidden_layers, nActions], beta = args.snn_beta, tSteps = args.snn_tSteps, DEBUG = args.debug)
    criticNetwork = qNetwork_SNN([stateSize[0], *args.hidden_layers, nActions], beta = args.snn_beta, tSteps = args.snn_tSteps, DEBUG = args.debug)
else:
    raise ValueError(f"Unknown architecture: {args.architecture}")

# Two models should have identical weights initially
criticNetwork.load_state_dict(actorNetwork.state_dict())

# TODO: Add gradient clipping to the optimizer for avoiding exploding gradients
# Suitable optimizer for gradient descent
optimActor = torch.optim.Adam(actorNetwork.parameters(), lr = args.learning_rate)
optimCritic = torch.optim.Adam(criticNetwork.parameters(), lr = args.learning_rate)

_networks = {
    "actorNetwork": actorNetwork,
    "criticNetwork": criticNetwork,
    "optimActor": optimActor,
    "optimCritic": optimCritic
}

args = vars(args) # Convert to dictionary
args["maxNumTimeSteps"] = 1000
args["action_space"] = actionSpace
args["env"] = env
args["stateSize"] = stateSize

args["agents"] = 1  # For now, only 1 agent is supported
args["uploadInfo"] = uploadInfo
args["run_save_path"] = runSavePath
args["timeStepsPerBatch"] = 4000
args["gamma"] = 0.95
args["clip"] = 0.2
args["nUpdatesPerIteration"] = 5

agent = PPO("HomePC", args, _networks)
agent.learn(150000)