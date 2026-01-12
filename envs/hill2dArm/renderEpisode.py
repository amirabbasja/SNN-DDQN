import pickle, torch, json, os, pprint
from hill2dArm import *

# Possible options: LundaLander-v3, hill2dArm
envName = "hill2dArm" 

# Location of the file that we get the data from
runLocation = "runs_data/2/"

# Read the environemnt information
with open(runLocation + "/conf.json") as _c:
    conf = json.load(_c)
envOptions = conf.get("env_options")
dt = envOptions.get("envParams").get("dt")
target = envOptions.get("target")
targetOffset = envOptions.get("targetOffset")

if(envName == "hill2dArm"):
    environemnt = hill2dArm(**envOptions)

# Read history from torch files
_hist = torch.load(next(os.path.join(runLocation, f) for f in os.listdir(runLocation) if f.endswith(".pth")), weights_only = False)
_hist = _hist.get("train_history")
i = 2
hist = _hist[i]
stepData = []
environemnt.reset(seed = hist.get("seed"))
saveDict = dict(environemnt.stepWiseParams)
saveDict.update(dict(environemnt.accumulatedRewards))
stepData.append(dict(saveDict))
theta = [np.array(environemnt.state)[0]]
thetaDot = [np.array(environemnt.state)[1]]
thetaRewards = [0]
omegaRewards = [0]
overallRewards = [0]
bicepsActivation = [0]
tricepsActivation = [0]
time = [0]
print("initial condition", hist.get("initialcondition"))
print("initial condition", environemnt.state)
print("seed", hist.get("seed"))
print("episode", hist.get("episode"))
print(len(_hist))
# lstActions = [4 for i in range(1000)]
# lstActions[0] = 0
# lstActions[3] = 0
# lstActions[5] = 0
# lstActions[6] = 0
# lstActions[7] = 0
# lstActions[8] = 0

counter = 0
for action in hist.get("actions"):
    _state, _reward, _terminated, _truncated, _info = environemnt.step(action)
    theta.append(np.array(environemnt.state)[0])
    thetaDot.append(np.array(environemnt.state)[1])

    saveDict = dict(environemnt.stepWiseParams)
    saveDict.update(dict(environemnt.accumulatedRewards))
    stepData.append(dict(saveDict))
    bicepsActivation.append(environemnt.bicepsActivation)
    tricepsActivation.append(environemnt.tricepsActivation)
    
    thetaRewards.append(_info.get("distanceReward"))
    omegaRewards.append(_info.get("velocityReward"))
    overallRewards.append(_reward)
    time.append(time[-1]+dt)
    if _terminated: break
    counter += 1
    if counter == 20: break

print("saved reward:",  hist.get("points"))
print("calculated reward:", sum(overallRewards))
dataDf = pd.DataFrame(stepData)

print("Availible columns to plot")
pprint.pprint(dataDf.columns.tolist())

colsToPlot = ["distanceReward","accumulatedDistanceReward", "rewardSum", "accumulatedRewardSum"]
dataDf = dataDf[colsToPlot]

# dataDf = dataDf.drop(["100WinRatio", "overallWinRatio"], axis = 1)
# dataDf = dataDf.rename({}, axis = 1)

dataDf["theta"] = theta
dataDf["thetaDot"] = thetaDot
dataDf["biceps Activation"] = bicepsActivation
dataDf["triceps activation"] = tricepsActivation

environemnt.createArmAnimation(
    time, theta, thetaDot, [target, target + targetOffset, target - targetOffset], runLocation + "/save.gif", thetaRewards, omegaRewards, dataDf
)