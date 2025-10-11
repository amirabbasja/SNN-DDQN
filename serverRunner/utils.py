import json, os, time, threading, asyncio, sys, json, time
from lightning_sdk import Machine, Studio
from functools import wraps

def _setCredentials(credentials):
    os.environ['LIGHTNING_API_KEY'] = credentials["apiKey"]
    os.environ['LIGHTNING_USER_ID'] = credentials["userID"]

def with_credentials(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        credentials = args[0] if args else kwargs.get("credentials")
        if not credentials:
            raise ValueError(f"Missing credentials for {func.__name__}")
        _setCredentials(credentials)
        return func(*args, **kwargs)
    return wrapper

@with_credentials
def stopStudio(credentials):
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    try:
        _studio.stop()
        print(f"Success on stopping studio {credentials['user']}")
    except Exception as e:
        print(f"Error stopping studio. Error: {e}")

@with_credentials
def startStudio(credentials):
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    _studio.start()

@with_credentials
def getStatus(credentials):
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    try:
        status = str(_studio.status)
        return status
    except Exception as e:
        return f"Error fetching status for {credentials['user']}: {e}"

@with_credentials
def uploadTrainingImages(credentials, botToken, chatId):
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    try:
        runCommand(credentials, f"python '/teamspace/studios/this_studio/serverRunner/uploadLatestTrainingData.py' --bot-token '{botToken}' --chat-id '{chatId}'")
    except Exception as e:
        print(f"Error uploading training images. Error: {e}")

@with_credentials
def runCommand(credentials, command):
    """
    Runs a designated studio with a specific command
    """
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    try:
        res = _studio.run(command)
        return res
    except:
        return False

@with_credentials
def startTraining(credentials, forceNewRun = False):
    """
    Starts a training bout by chekcing the status of the studio and running the command

    Args:
        credentials (dict): Dictionary containing the studio credentials
        forceNewRun (bool, optional): If True, forces a new run by adding the --forcenewrun 
            flag to the command. Defaults to False.
    """

    _status = getStatus(credentials)
    if(_status == "Stopping"):
        while True:
            print("Studio is stopping. Waiting for it to stop...")
            _status = getStatus(credentials)
            if(_status != "Stopping"):
                break
            time.sleep(30)
    
    if(_status != "Stopped"):
        try:
            stopStudio(credentials)
            time.sleep(10)
        except:
            pass
    
    # Start the studio
    startStudio(credentials)
    time.sleep(10)

    # If instructed to force a new run, modify the command and add a flag
    if(forceNewRun):
        credentials["commandToRun"] = credentials["commandToRun"] + " --forcenewrun"

    # Run the command
    runCommand(credentials, credentials["commandToRun"])