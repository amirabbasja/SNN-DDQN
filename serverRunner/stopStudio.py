import sys, os, json
from utils import *

json_string = sys.argv[1] if len(sys.argv) > 1 else "{}"
param = json.loads(json_string)

stopStudio(param)
