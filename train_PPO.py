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

# Deserialize the JSON strings into dictionaries
args.network_options = json.loads(args.network_options)
args.algorithm_options = json.loads(args.algorithm_options)

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
env = gym.make(args.env)
state, info = env.reset() # Get a sample state of the environment 
stateSize = env.observation_space.shape # Number of variables to define current step 
nActions = env.action_space.n # Number of actions 
actionSpace = np.arange(nActions).tolist() 

# Make the model objects
if args.network == "ann":
    actorNetwork = qNetwork_ANN([stateSize[0], *args.network_options["hidden_layers_actor"], nActions])
    criticNetwork = qNetwork_ANN([stateSize[0], *args.network_options["hidden_layers_critic"], 1])
elif args.network == "snn":
    actorNetwork = qNetwork_SNN([stateSize[0], *args.network_options["hidden_layers_actor"], nActions], beta = args.network_options["snn_beta"], tSteps = args.network_options["snn_tSteps"], DEBUG = args.debug)
    criticNetwork = qNetwork_SNN([stateSize[0], *args.network_options["hidden_layers_critic"], 1], beta = args.network_options["snn_beta"], tSteps = args.network_options["snn_tSteps"], DEBUG = args.debug)
else:
    raise ValueError(f"Unknown network: {args.network}")

# TODO: Add gradient clipping to the optimizer for avoiding exploding gradients
# Suitable optimizer for gradient descent
optimActor = torch.optim.Adam(actorNetwork.parameters(), lr = args.algorithm_options["learning_rate"])
optimCritic = torch.optim.Adam(criticNetwork.parameters(), lr = args.algorithm_options["learning_rate"])

_networks = {
    "actorNetwork": actorNetwork,
    "criticNetwork": criticNetwork,
    "optimActor": optimActor,
    "optimCritic": optimCritic
}

args = vars(args) # Convert to dictionary
args["action_space"] = actionSpace
args["env"] = env
args["stateSize"] = stateSize

args["uploadInfo"] = uploadInfo
args["run_save_path"] = runSavePath

agent = PPO(os.getenv("session_name"), args, _networks)
agent.learn()