# General introduction to repo

This repo is for training an RL agent using SNNs. The training is done using the [OpenAI Gym](https://gym.openai.com/) environment.  
We have deployed a DDQN architecture for training the agent. Both online and critique networks are SNNs.  
  
To run the code, two important files are required which are not added to repo and should be kep sectret. These files should contain following parameters:  
1. *.env*  
    ```
    session_name = <name of the session>
    huggingface_read = <read token for huggingFace>
    huggingface_write = <write token for huggingFace>
    repo_ID = <huggingFace repo ID>
    code_base_link = <github repo ID to update the files to the latest version before training>
    python_venv_path = <path to the venv exe file>
    ```
2. *conf.json*
    ```
    {
        "name": "SNN_DDQN",
        "continue_run": true,
        "agents": 1,
        "hidden_layers": [64, 64],
        "learning_rate": 0.0004,
        "decay": 0.998,
        "batch": 100,
        "gamma": 0.995,
        "extra_info": "",
        "max_run_time": 12600, 
        "stop_learning_at_win_percent": 80,
        "train_max_time": 12600,
        "upload_to_cloud": false,
        "local_backup": true,
        "architecture": "snn",
        "snn_tSteps": 25,
        "snn_beta": 0.95,
        "debug": true
    }
    ```
3. *studios.json*
    ```
    {
        "studio_1": {
            "user": "testName", <-- Session name>
            "apiKey": "...", <-- lightning ai api key
            "userID": "...", <-- lightning ai user ID
            "teamspaceName": "Vision-model", <-- lightning ai api teamspace name
            "studioName": "paper_no.6", <-- lightning ai api studio name
            "commandToRun": "python /teamspace/studios/this_studio/run.py" <-- Command to run
        },
    }
    ```
  
  
---
  
# Description of the files and folders

1. *serverRunner/:* A directory for running the automated training runs. This folder requires node.js to be installed on machine. Run the file *trainingRunner.js* to start the training which will get all provided **lightning AI** studios (located in *studios.json*) to run, run them with a pre-defined delay with a 4 hour delay between each batch of run. This will help with staying in the free hours of the platform (First 4 hours are free). The javascript file runs *sendReq.py* to run the studios which sends a command to run the `commandToRun` key in studios.json. The script is coded so to work in the scheme of *fire and forget* and after making a request to run the command in the studio, nothing else is checked with it and after a set amount of time has passed, *trainingRunner.js* will kill the child process. So NO feedback is given to the user and the training is done in the background. This should be done seperately on the train.py file.

2. *models.py:* A file containing necessary calsses to define the network architectures. Currently SNNs (Spiking Neural Networks) and classic ANNs are supported.

3. *run.py:* Tasked with running the *train.py* file with necessary arguments, acquired from *.env* and *conf.json* files. Another **VERY** important task of this file is to keep the entire repository up-to-date with the latest version of the code pointed in the `code_base_link` key of environemnt file. When running, checks to see if all packages are installed, then downloads and replaces the files with the latest version. If the *run.py* file has been changed as well, first it updates itself to the latest version in the repository, then restarts (Runs itself as a subprocess). I deployed this functionality to reduce the amount of effort required to keep the code base up-to-date in multiple servers. Running this file without any arguments, defaults to running *train.py*; however, if an argument is passed as well, it should be a file location to run instead of *train.py*. Also, if this file is ran with *"--forecenewrun"* flag, training will not continue from latest progress, regardless of *cong.json* file. There a re multiple ways to use this file which are explained below:
    * Run the file stand alone using `python run.py`. This results in applying the latest version of the code to the repository and then running *train.py* with the arguments provided in *.env* and *conf.json* files.  
    Running the file with a `--forcenewrun` flag will force the training to start from scratch and not contunue the latest run (If "continue_run" is set to true in *conf.json* file), regardless of *conf.json* file.
    * Run the file with a `--forceconfig` flag which with doing so, the app will disregard all of the configs present  in the *conf.json* file. With passing the `--forceconfig` flag, the app expects you to pass the entire configuration as a JSON string with `--config <JSON string>` argument.

4. *train.py:* Performs the training process. Shouldn't be ran stand-alone.

5. *utils.py:* Houses the utility functions necessary to keep the entire repo working.