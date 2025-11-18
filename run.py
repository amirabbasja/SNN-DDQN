# Runs the training script with desired parameters
import subprocess, os, time, sys, json, torch, requests, base64
from dotenv import load_dotenv
from urllib.parse import urlparse

def download_repo_files(repo_name, branch='main', exclude_files=None, local_dir=None):
    """
    Downloads files from a GitHub repository one by one (excluding specified files).
    
    Args:
        repo_name (str): Repository name in format 'owner/repo' (e.g., 'octocat/Hello-World')
        branch (str): Branch name to download from (default: 'main')
        exclude_files (list): List of filenames to exclude (default: ['run.py'])
        local_dir (str): Local directory to save files (default: repo name)
    
    Returns:
        dict: Summary of download results
    """
    if exclude_files is None:
        exclude_files = ['run.py']
    
    if local_dir is None:
        local_dir = repo_name.split('/')[-1]
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # GitHub API URLs
    api_base = f"https://api.github.com/repos/{repo_name}"
    tree_url = f"{api_base}/git/trees/{branch}?recursive=1"
    
    results = {
        'downloaded': [],
        'skipped': [],
        'errors': [],
        'total_files': 0
    }
    
    try:
        # Get repository tree
        print(f"Fetching repository tree for {repo_name}...")
        tree_response = requests.get(tree_url)
        tree_response.raise_for_status()
        tree_data = tree_response.json()
        
        if 'tree' not in tree_data:
            raise Exception("Repository tree not found. Check if repo exists and is public.")
        
        files = [item for item in tree_data['tree'] if item['type'] == 'blob']
        results['total_files'] = len(files)
        
        print(f"Found {len(files)} files in repository")
        
        for file_item in files:
            file_path = file_item['path']
            file_name = os.path.basename(file_path)
            
            # Skip excluded files
            if file_name in exclude_files:
                print(f"Skipping excluded file: {file_path}")
                results['skipped'].append(file_path)
                continue
            
            try:
                # Download individual file
                file_url = f"{api_base}/contents/{file_path}?ref={branch}"
                file_response = requests.get(file_url)
                file_response.raise_for_status()
                file_data = file_response.json()
                
                # Create directory structure
                local_file_path = os.path.join(local_dir, file_path)
                local_file_dir = os.path.dirname(local_file_path)
                if local_file_dir:
                    os.makedirs(local_file_dir, exist_ok=True)
                
                # Decode and save file content
                if file_data.get('encoding') == 'base64':
                    content = base64.b64decode(file_data['content'])
                    with open(local_file_path, 'wb') as f:
                        f.write(content)
                else:
                    # Handle other encodings or plain text
                    content = file_data.get('content', '')
                    with open(local_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                print(f"Downloaded: {file_path}")
                results['downloaded'].append(file_path)
                
                # Small delay to be respectful to GitHub API
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Error downloading {file_path}: {str(e)}"
                print(error_msg)
                results['errors'].append(error_msg)
                continue
    
    except Exception as e:
        print(f"Error accessing repository: {str(e)}")
        results['errors'].append(f"Repository access error: {str(e)}")
        return results
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"Total files found: {results['total_files']}")
    print(f"Successfully downloaded: {len(results['downloaded'])}")
    print(f"Skipped (excluded): {len(results['skipped'])}")
    print(f"Errors: {len(results['errors'])}")
    
    return results

def update_run_file(repo_name, branch='main'):
    """
    Downloads and updates the run.py file itself from the repository.
    Returns True if the file was updated, False if no update was needed.
    """
    current_file = os.path.abspath(__file__)
    temp_file = current_file + ".tmp"
    
    api_base = f"https://api.github.com/repos/{repo_name}"
    file_url = f"{api_base}/contents/run.py?ref={branch}"
    
    try:
        # Download the latest run.py
        file_response = requests.get(file_url)
        file_response.raise_for_status()
        file_data = file_response.json()
        
        if file_data.get('encoding') == 'base64':
            new_content = base64.b64decode(file_data['content'])
        else:
            new_content = file_data.get('content', '').encode('utf-8')
        
        # Read current file content
        with open(current_file, 'rb') as f:
            current_content = f.read()
        
        # Compare content - decode to strings for better diff analysis
        new_content_str = new_content.decode('utf-8', errors='ignore')
        current_content_str = current_content.decode('utf-8', errors='ignore')
        
        # Remove empty lines and whitespace for comparison
        new_clean = [line.strip() for line in new_content_str.splitlines() if line.strip()]
        current_clean = [line.strip() for line in current_content_str.splitlines() if line.strip()]
        
        # Check if there are meaningful differences
        has_meaningful_differences = new_clean != current_clean
        
        if not has_meaningful_differences:
            print("run.py is already up to date (no meaningful differences).")
            return False
        
        print("Meaningful differences found in run.py. Updating...")
        
        # Write to temporary file first
        with open(temp_file, 'wb') as f:
            f.write(new_content)
        
        # Replace the current file
        os.replace(temp_file, current_file)
        print("run.py has been updated successfully with meaningful changes.")
        
        # Restart the script with the updated version
        # print("Restarting with updated version...")
        subprocess.run(sys.executable, [sys.executable, current_file] + sys.argv[1:])
        
    except Exception as e:
        print(f"Error updating run.py: {str(e)}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

def isValidPath(string):
    """
    Checks to see if a string is a valid file path

    Args: 
        string (str): The string to check
    Returns:
        bool: True if valid path, False otherwise
    """
    if not string:
        return False
    try:
        # Normalize the path to check its validity
        os.path.normpath(string)
        return True
    except (ValueError, OSError):
        return False

def sendTelegramMessage(bot_token, chat_id, message):
    
    """Send a message to a Telegram user/chat"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    return response.json()

# # Install packages
# subprocess.check_call([sys.executable, "-m", "pip", "install", "swig"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'gymnasium[box2d]'])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "snntorch"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

# IF facing box2d not found errors, despite installing it, try following two lines:
# conda install -c conda-forge swig box2d-py
# pip install "gymnasium[box2d]"

# Before everything, update to the latest codebase
load_dotenv()
if os.getenv("code_base_link") != "." and os.getenv("code_base_link") != None and not "--skipupdate" in sys.argv:
    try:
        print("Updating codebase...")
    
        # First update run.py itself
        print("Checking for run.py updates...")
        if update_run_file(os.getenv("code_base_link"), "master"):
            # If updated, the script will restart and we won't reach here
            pass
        else:
            # Then update other files (excluding run.py)
            download_repo_files(os.getenv("code_base_link"), "master", ["run.py"], os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"Error updating codebase: {str(e)}")

if "--onlyupdate" in sys.argv:
    print("Only updated codebase. Skipping training...")
    exit()

# Check to see if any new arguments were passed
forceNewRun = False
if "--forcenewrun" in sys.argv:
    forceNewRun = True
    sys.argv.remove("--forcenewrun")

# Check to see if new parameters are forced, if not, default to conf.json file
if "--forceconfig" in sys.argv:
    print("Forcing new configuration...")

    if "--config" in sys.argv:
        configIndex = sys.argv.index("--config")
        configString = sys.argv[configIndex + 1]

        try:
            data = json.loads(configString)
        except Exception as e:
            raise Exception("Invalid JSON format in --config parameter. Configuration should be passed exactly after --config flag. Error: " + e.__str__())
    else:
        raise Exception("Expected --config when --forceconfig is passed")
else:
    # Open and read run configuration
    assert os.path.exists("conf.json"), "conf.json file doesn't exist"
    try:
        with open('conf.json', 'r') as file:
            data = json.load(file)
    except Exception:
        raise Exception("Error reading conf.json file. Please check the file format.")

# Choose script path based on algorithm
if data["algorithm"] == "PPO":
    scriptPath = "./train_PPO.py"
elif data["algorithm"] == "DDQN":
    scriptPath = "./train_DDQN.py"
else:
    raise ValueError(f"Unknown algorithm specified in conf.json: {data['algorithm']}")

startTime = time.time()
endTime = startTime + data["train_max_time"]  # 3.5 hours
maxRunTime = data["max_run_time"]

if(os.getenv("telegram_chat_id") and os.getenv("telegram_bot_token") and os.getenv("telegram_bot_token") != "."):
    sendTelegramMessage(os.getenv("telegram_bot_token"), os.getenv("telegram_chat_id"), f"Training started for session {os.getenv('session_name')}")

# Replace the conf.json contents with the new parameters
if "--forceconfig" in sys.argv:
    with open("conf.json", 'w') as f:
        json.dump(data, f, indent=4)

argsDict = {} # Save the runs argument
trainingEpoch = 1
while time.time() < endTime:
    # Check for "finished" value in the config file
    with open('conf.json', 'r') as file:
        _updatedJson = json.load(file)
        if not "finished" in _updatedJson:
            print("Finished key not found in conf.json. Please check the file format.")
            break

        if _updatedJson["finished"]:
            print("The config has been marked as finished. Exiting...")
            break

    # Run parameters
    if data["algorithm"] == "DDQN":
        argsDict = {
            "name": data["name"],
            "env": data["env"],
            "env_options": data["env_options"],
            "algorithm": data["algorithm"],
            "algorithm_options": data["algorithm_options"],
            "network": data["network"],
            "network_options": data["network_options"],
            "continue_run": data["continue_run"], # If --forcenewrun was passed, override config
            "agents": data["agents"],
            "extra_info": data["extra_info"],
            "max_run_time": data["max_run_time"], # In seconds
            "stop_learning_at_win_percent": data["stop_learning_at_win_percent"],
            "upload_to_cloud": data["upload_to_cloud"],
            "local_backup": data["local_backup"],
            "debug": data["debug"],
            "train_finish_timestamp": endTime,
            "stop_condition": data["stop_condition"],
            "finished": data["finished"]
        }
    elif  data["algorithm"] == "PPO":
        argsDict = {
            "name": data["name"],
            "env": data["env"],
            "env_options": data["env_options"],
            "algorithm": data["algorithm"],
            "algorithm_options": data["algorithm_options"],
            "network_actor": data["network_actor"],
            "network_actor_options": data["network_actor_options"],
            "network_critic": data["network_critic"],
            "network_critic_options": data["network_critic_options"],
            "continue_run": data["continue_run"], # If --forcenewrun was passed, override config
            "agents": data["agents"],
            "extra_info": data["extra_info"],
            "max_run_time": data["max_run_time"], # In seconds
            "stop_learning_at_win_percent": data["stop_learning_at_win_percent"],
            "upload_to_cloud": data["upload_to_cloud"],
            "local_backup": data["local_backup"],
            "debug": data["debug"],
            "train_finish_timestamp": endTime,
            "stop_condition": data["stop_condition"],
            "finished": data["finished"]
        }
    else:
        raise ValueError(f"Unknown algorithm specified in conf.json: {data['algorithm']}")

    # Override continue_run if needed
    if forceNewRun:
        argsDict["continue_run"] = False

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
        
        if name == "debug":
            scriptArgs.extend([f"--debug"]) if value else None
            continue
        
        if name == "algorithm_options":
            scriptArgs.extend([f"--algorithm_options", json.dumps(value)])
            continue
        
        if name == "network_options":
            scriptArgs.extend([f"--network_options", json.dumps(value)])
            continue
        
        if name == "network_actor_options":
            scriptArgs.extend([f"--network_actor_options", json.dumps(value)])
            continue
        
        if name == "network_critic_options":
            scriptArgs.extend([f"--network_critic_options", json.dumps(value)])
            continue
        
        if name == "stop_condition":
            scriptArgs.extend([f"--stop_condition", json.dumps(value)])
            continue

        if name == "finished":
            scriptArgs.extend([f"--finished"]) if value else None
            continue
        
        scriptArgs.extend([f"--{name}", str(value)])

    if "--forceconfig" in sys.argv:
        scriptArgs.extend([f"--forcedconfig", json.dumps(data)])
    
    venvPath = str(os.getenv("python_venv_path"))

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

if(os.getenv("telegram_chat_id") and os.getenv("telegram_bot_token") and os.getenv("telegram_bot_token") != "."):
    sendTelegramMessage(os.getenv("telegram_bot_token"), os.getenv("telegram_chat_id"), f"Training finished for session {os.getenv('session_name')}. Took {((time.time() - startTime)/3600):.2f} hours.")

print(f"Reached the maximum run time. Trained {trainingEpoch} epochs")
