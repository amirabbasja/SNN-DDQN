import json, os, time, threading, asyncio, sys, json, time
from lightning_sdk import Machine, Studio

def _setCredentials(credentials):
    os.environ['LIGHTNING_API_KEY'] = credentials["apiKey"]
    os.environ['LIGHTNING_USER_ID'] = credentials["userID"]

def stopStudio(credentials):
    _setCredentials(credentials)
    
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    try:
        _studio.stop()
    except:
        print("Error stopping studio")

def startStudio(credentials):
    _setCredentials(credentials)
    
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    _studio.start()

def getStatus(credentials):
    _setCredentials(credentials)
    
    _studio = Studio(
        credentials["studioName"],       # studio name
        credentials["teamspaceName"],    # teamspace name
        user = credentials["user"],
        create_ok = False
    )
    
    return str(_studio.status)

def runCommand(credentials, command):
    """
    Runs a designated studio with a specific command
    """
    _setCredentials(credentials)
    
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

def startTraining(credentials):
    """
    Starts a training bout by chekcing the status of the studio and running the command
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

    startStudio(credentials)
    time.sleep(10)

    runCommand(credentials, credentials["commandToRun"])