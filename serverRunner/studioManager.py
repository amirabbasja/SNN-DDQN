import sys, os, json
from utils import *

params = json.loads(sys.argv[1])
action = params.get("action", "")
credentials = json.loads(params.get("credentials", {}))
forceNewRun = params.get("forceNewRun", False)

if action == "stop_single":
    stopStudio(credentials)
elif action == "start_single":
    startStudio(credentials)
elif action == "train_single":
    startTraining(credentials, forceNewRun)
elif action == "status_single":
    getStatus(credentials)
elif action == "training_stat":
    uploadTrainingImages(credentials, params["botToken"], params["chatId"])
else:
    print("No valid action provided")