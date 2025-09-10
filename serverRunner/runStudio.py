# Runs lightning.ai studios by acquiring their inforamtion from studios.json file
import json, sys, json
from utils import *

json_string = sys.argv[1] if len(sys.argv) > 1 else "{}"
param = json.loads(json_string)

startTraining(param)
