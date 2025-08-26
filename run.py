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

# Before everything, update to the latest codebase
load_dotenv()
if os.getenv("code_base_link") != "." and os.getenv("code_base_link") != None:
    print("Updating codebase...")
    download_repo_files(os.getenv("code_base_link"), "master", ["run.py"], os.path.dirname(os.path.abspath(__file__)))

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
        "local_backup": data["local_backup"],
        "architecture": data["architecture"],
        "snn_tSteps": data["snn_tSteps"],
        "snn_beta": data["snn_beta"]
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

    venvPath = str(os.getenv("python_venv_path"))
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