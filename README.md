# General introduction to repo

This repo is for training an RL agent using SNNs. The training is done using the [OpenAI Gym](https://gym.openai.com/) environment.  
We have deployed a DDQN architecture for training the agent. Both online and critique networks are SNNs.  
  
To run the code, two important files are required which are not added to repo and should be kep sectret. These files should contain following parameters:  
1. `.env`  
  `
    session_name = <name of the session>
    huggingface_read = <read token for huggingFace>
    huggingface_write = <write token for huggingFace>
    repo_ID = <huggingFace repo ID>
    code_base_link = <github repo ID to update the files to the latest version before training>
    python_venv_path = <path to the venv exe file>
  `
2. `conf.json`
  `
    {
        "name": "SNN_DDQN",
        "continue_run": false,
        "agents": 1,
        "hidden_layers": [64, 64],
        "learning_rate": 0.0004,
        "decay": 0.998,
        "batch": 100,
        "gamma": 0.995,
        "extra_info": "",
        "max_run_time": 2700, 
        "train_max_time": 12600,
        "upload_to_cloud": false,
        "local_backup": false,
        "architecture": "snn",
        "snn_tSteps": 25,
        "snn_beta": 0.95
    }
  `