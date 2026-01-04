import pickle, torch, json, os
from envs.hill2dArm.hill2dArm import *

# Possible options: LundaLander-v3, hill2dArm
envName = "hill2dArm" 

# Location of the file that we get the data from
runLocation = "runs_data/21/"

# Read the environemnt information
with open(runLocation + "/conf.json") as _c:
    conf = json.load(_c)
envOptions = conf.get("env_options")
dt = envOptions.get("envParams").get("dt")
target = envOptions.get("target")
targetOffset = envOptions.get("targetOffset")

with open(runLocation + "/actions.pkl", "rb") as f:
    data = pickle.load(f)

if(envName == "hill2dArm"):
    environemnt = hill2dArm(**envOptions)

# Read history from torch files
_hist = torch.load(next(os.path.join(runLocation, f) for f in os.listdir(runLocation) if f.endswith(".pth")), weights_only = False)
_hist = _hist.get("train_history")
i = 0
hist = _hist[i]
environemnt.reset(seed=data[i].get("seed"))
theta = [np.array(environemnt.state)[0]]
thetaDot = [np.array(environemnt.state)[0]]
thetaRewards = [0]
omegaRewards = [0]
overallRewards = [0]
time = [0]
print("seed", hist.get("seed"))
print("seed", data[i].get("seed"))
print("episode", hist.get("episode"))
print("episode", data[i].get("episode"))
print(len(_hist))
print(len(data))
for action in data[i].get("actions"):
    _state, _reward, _terminated, _truncated, _info = environemnt.step(action)
    theta.append(np.array(environemnt.state)[0])
    thetaDot.append(np.array(environemnt.state)[1])

    thetaRewards.append(_info.get("distanceReward"))
    omegaRewards.append(_info.get("velocityReward"))
    overallRewards.append(_reward)
    time.append(time[-1]+dt)
print("reward:", sum(overallRewards))
environemnt.createArmAnimation(
    time, theta, thetaDot, [target, target + targetOffset, target - targetOffset], runLocation + "/save.gif", thetaRewards, omegaRewards
)