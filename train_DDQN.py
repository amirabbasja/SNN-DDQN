from models import *
from huggingface_hub import HfApi, login
import os
from utils import *
import numpy as np
import gymnasium as gym
import torch
from DDQN import DDQN

# Parse model arguments
parser = modelParamParser()
args, unknown = parser.parse_known_args()

# Deserialize the JSON strings into dictionaries
args.env_options = json.loads(args.env_options)
args.network_options = json.loads(args.network_options)
args.algorithm_options = json.loads(args.algorithm_options)
args.stop_condition = json.loads(args.stop_condition)
try:
    # IF extra info is a dict, decode it
    args.extra_info = json.loads(args.extra_info)
except:
    # If its a comment, keep it as is
    args.extra_info = args.extra_info if type(args.extra_info) == str else args.extra_info

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

# Impose finished condition
if args.finished:
    print("Training is already finished. Exiting...")
    exit()

continueLastRun = args.continue_run
_, runSavePath = get_next_run_number_and_create_folder(continueLastRun, args)

if "--forcedconfig" in unknown:
    # IF forced a new config, don't save the default config, save the forced one
    _config = json.loads(unknown[unknown.index("--forcedconfig") + 1])

    # Save the file
    with open(os.path.join(runSavePath, "conf.json"), 'w') as f:
        json.dump(_config, f, indent=4) 
else:
    # Copy the config file to the run folder
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "conf.json"), os.path.join(runSavePath, "conf.json"))

# Make the environment
_customEnvironment = False
if args.env in ["LunarLander-v3"]:
    try:
        envName = args.env
        env = gym.make(args.env)
        state, info = env.reset() # Get a sample state of the environment 
        stateSize = env.observation_space.shape # Number of variables to define current step 
        nActions = env.action_space.n # Number of actions 
        actionSpace = np.arange(nActions).tolist() 
    except:
        raise ValueError(f"Error when making the environment {args.env}")
else:
    # Custom environment - Look for environment class in ./envs/<name>/<name>.py:<name>
    # where <name> is the name of the custom environemnt
    if not os.path.isfile(f"./envs/{args.env}/{args.env}.py"):
        raise Exception(f"Custom environemnt {args.env} not found in /envs/{args.env}/{args.env}.py")
    
    # Import the environment class
    envClass = checkAndImportClass(f"envs.{args.env}.{args.env}", args.env)
    if envClass is None:
        raise ValueError(f"Error when importing the custom environment class {args.env}")
    try:
        envName = args.env
        env = envClass(**args.env_options)
        state, info = env.reset() # Get a sample state of the environment 
        stateSize = env.nObservationSpace # Number of variables to define current step 
        nActions = env.nActionSpace # Number of actions 
        actionSpace = np.arange(nActions).tolist() 
        _customEnvironment = True
    except Exception as e:
        raise ValueError(f"Error when making the custom environment {args.env}: {str(e)}")

# Handle the necessary env_options
if args.env_options.get("observationNormalization", False):
    normalizationFunctionName = args.env_options.get("normalizationFunction", None)

    if not normalizationFunctionName:
        raise ValueError("Normalization function not specified in env_options despite observationNormalization being True")

    if normalizationFunctionName == "RunningMeanStd":
        # TODO: Add a enum for supported normalization functions
        # Do nothing
        pass
    else:
        if _customEnvironment:
            if envClass and not hasattr(envClass, normalizationFunctionName):
                raise ValueError(f"Normalization function {normalizationFunctionName} not found in custom environment {args.env}")

# Make the model objects
if args.network == "ann":
    qNetwork_model = qNetwork_ANN([stateSize[0], *args.network_options["hidden_layers"], nActions])
    targetQNetwork_model = qNetwork_ANN([stateSize[0], *args.network_options["hidden_layers"], nActions])
elif args.network == "snn":
    qNetwork_model = qNetwork_SNN([stateSize[0], *args.network_options["hidden_layers"], nActions], beta = args.network_options["snn_beta"], tSteps = args.network_options["snn_tSteps"], DEBUG = args.debug)
    targetQNetwork_model = qNetwork_SNN([stateSize[0], *args.network_options["hidden_layers"], nActions], beta = args.network_options["snn_beta"], tSteps = args.network_options["snn_tSteps"], DEBUG = args.debug)
else:
    raise ValueError(f"Unknown network: {args.network}")

# Two models should have identical weights initially
targetQNetwork_model.load_state_dict(qNetwork_model.state_dict())

# TODO: Add gradient clipping to the optimizer for avoiding exploding gradients
# Suitable optimizer for gradient descent
optimizer_main = torch.optim.Adam(qNetwork_model.parameters(), lr = args.algorithm_options["learning_rate"])
optimizer_target = torch.optim.Adam(targetQNetwork_model.parameters(), lr = args.algorithm_options["learning_rate"])

_networks = {
    "qNetwork_model": qNetwork_model,
    "targetNetwork_model": targetQNetwork_model,
    "optimizer_main": optimizer_main,
    "optimizer_target": optimizer_target
}

args = vars(args) # Convert to dictionary
args["action_space"] = actionSpace
args["env"] = env
args["envName"] = envName
args["stateSize"] = stateSize

if args["algorithm"] != "DDQN":
    raise ValueError(f"Algorithm should be DDQN, not {args['algorithm']}")

args["uploadInfo"] = uploadInfo
args["run_save_path"] = runSavePath

agent = DDQN(os.getenv("session_name"), args, _networks)
agent.train()