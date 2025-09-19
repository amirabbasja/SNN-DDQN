import sys, os, json
from utils import *

action = sys.argv[1] if len(sys.argv) > 1 else "{}"

json_string = sys.argv[2] if len(sys.argv) > 2 else "{}"
param = json.loads(json_string)

forceNewRun = sys.argv[3] if len(sys.argv) > 3 else False

if action == "stop_single":
    stopStudio(param)
elif action == "start_single":
    startStudio(param)
elif action == "train_single":
    startTraining(param, forceNewRun)
elif action == "status_single":
    getStatus(param)
else:
    print("No valid action provided")