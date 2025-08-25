# Runs the training script with desired parameters
import subprocess, os, time, sys, json,torch
from utils import updateCodeBase
from dotenv import load_dotenv

# Before everything, update to the latest codebase
load_dotenv()
if os.getenv("code_base_link"):
    print("Updating codebase...")
    updateCodeBase(os.getenv("code_base_link"))

# Open and read run configuration
assert os.path.exists("conf.json"), "conf.json file doesn't exist"
with open('conf.json', 'r') as file:
    data = json.load(file)

startTime = time.time()
endTime = startTime + data["train_max_time"]  # 3.5 hours
maxRunTime = data["max_run_time"]  # 45 min

trainingEpoch = 1
while time.time() < endTime:
    # Run parameters
    argsDict = {
        "name": data["name"],
        "continue_run": data["continue_run"],
        "agents": data["agents"],
        "hidden_layers": data["hidden_layers"],
        "learning_rate": data["learning_rate"],
        "decay": data["decay"],
        "batch": data["batch"],
        "gamma": data["gamma"],
        "extra_info": "",
        "max_run_time": data["max_run_time"], # In seconds
        "upload_to_cloud": data["upload_to_cloud"],
        "local_backup": data["local_backup"]
    }

    # For passing the args to the script
    scriptArgs = []
    for name, value in argsDict.items():
        if name == "continue_run":
            scriptArgs.extend([f"--continue_run"]) if value else None
            continue
        
        if name == "upload_to_cloud":
            scriptArgs.extend([f"--upload_to_cloud"]) if value else None
            continue
        
        if name == "local_backup":
            scriptArgs.extend([f"--local_backup"]) if value else None
            continue
        
        if name == "hidden_layers":
            scriptArgs.extend([f"--{name}"] + [str(l) for l in value])
        else:
            scriptArgs.extend([f"--{name}", str(value)])

    venvPath = "C:/Users/Spino/Desktop/Codes/Projects/.motherEnv/Scripts/python.exe"
    scriptPath = "./train.py"

    command = [venvPath, scriptPath] + scriptArgs
    
    try:
        # Set environment to force unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout
            text=True,
            bufsize=0,
            universal_newlines=True,
            env=env
        )

        # Read output line by line in real-time
        for i, line in enumerate(iter(process.stdout.readline, '')):
            print(line, end='', flush=True)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Script failed with return code {process.returncode}")
    except FileNotFoundError:
        print(f"Virtual environment Python or script not found. Check paths: {venvPath}, {scriptPath}")
    except Exception as e:
        print(f"Error running script: {e}")
    
    trainingEpoch += 1

print(f"Reached the maximum run time. Trained {trainingEpoch} epochs")