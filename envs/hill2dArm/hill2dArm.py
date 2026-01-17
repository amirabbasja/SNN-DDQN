from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math, os
import pandas as pd
from IPython.display import HTML, display
deg = np.pi / 180

# PART 01- Musculoskeletal Parameters ======================================================================
# ==========================================================================================================
@dataclass
class MuscleParams:
    # Parameters for Hill's muscle model
    Length_min, Length_max= 0.55, 1.75 # The length Range that muscle excerts force by varying its length

    # Parameters for Hill's muscle model - Biceps
    optimal_fiber_length_Biceps: float = 0.17  # meters
    max_contraction_velocity_Biceps: float = .85  # m/s
    max_isometric_force_Biceps: float = 300  # Newtons
    pennation_angle_Biceps: float = 0  # radians
    tendon_slack_length_Biceps: float = 0.15  # meters

    # Parameters for Hill's muscle model - Triceps
    optimal_fiber_length_Triceps: float = 0.19  # meters
    max_contraction_velocity_Triceps: float = .95  # m/s
    max_isometric_force_Triceps: float = 300  # Newtons
    pennation_angle_Triceps: float = 0  # radians
    tendon_slack_length_Triceps: float = 0.14  # meters

    # Anatomical Parameters
    tendon_start_to_elbow: float = 0.3  # meters (adult male)
    elbow_to_biceps_end = .05 # meters
    elbow_to_triceps_end = .05 # meters
    forearm_length = 0.28 # meters, for men
    upper_arm_length = 0.32 # meters

# PART 02- Gymnasium environmert Construction ==============================================================
# ==========================================================================================================
class hill2dArm(gym.Env):
    """
    --V2--

    Makes a gymnasium environment for a simple human forearm movement.
    The upper arm is taken to be fixed and the forearm is controlled 
    via biceps and triceps muscles. Each muscle has a certain activation
    level which results in arm's movement. By default, the target for 
    angular velocity is 0 rad/s

    Version change log:
    v3: Add angular velocity to the target as well TODO
    v2: Now when taking a step, or resetting, it returns the relative state (to the targets) instead of the state itself
    v1: The initial version
    """
    def __init__(self, initialConditions, rewardWeights, envParams, target, targetOffset):
        super().__init__()

        # Define the target
        if target < envParams["thetaMin"] or envParams["thetaMax"] < target:
            raise ValueError(f"Target must be between thetaMin and thetaMax (In Radians). target: {target}, thetaMin: {envParams['thetaMin']}, thetaMax: {envParams['thetaMax']}")
        if envParams["thetaMax"] <= envParams["thetaMin"]:
            raise ValueError(f"Theta max should be higher than theta min. You entered thetaMin: {envParams['thetaMin']}, thetaMax: {envParams['thetaMax']}")
        self.target = target

        # Define the target offset
        self.targetOffset = targetOffset

        # Environemnt params
        self.envParams = envParams

        # The action space: 
        # 1) biceps activation, 
        # 2) triceps activation; 
        # both are between 0 and 1 and independent of eachother
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (5,), dtype = np.float32)
        self.nActionSpace = 5
        
        # Set target velocity
        self.targetVelocity = 0

        # The observation space:
        # 1) angular difference from target angle, 
        # 2) angular velocity difference from target angular velocity
        self.observation_space = spaces.Box(
            low = np.array([np.float32(self.envParams["thetaMin"] - self.target), np.float32(self.envParams["omegaMin"] - self.targetVelocity)]), 
            high = np.array([np.float32(self.envParams["thetaMax"] - self.target), np.float32(self.envParams["omegaMax"] - self.targetVelocity)]) , 
            shape = (2,), 
            dtype = np.float32
        )
        self.nObservationSpace = (2,)

        # Used for normalizing the rewards, the ranges consist of the max/min of the acceptable range to the target
        self._thetaRange = [abs(self.envParams["thetaMin"] - self.target), abs(self.envParams["thetaMax"] - self.target)]
        self._omegaRange = [abs(self.envParams["omegaMin"] - self.targetVelocity), abs(self.envParams["omegaMax"] - self.targetVelocity)]

        # Initialize the state
        if not isinstance(initialConditions, np.ndarray):
            if(initialConditions == None):
                self.state = np.array([0.0, 0.0])
            else:
                self.state = np.array(initialConditions)
        else:
            self.state = np.array(initialConditions)
            
        # Muscle parameters
        self.muscleParams = MuscleParams()

        # Define step number
        self._stepNum = 0
        self.envSave = None

        # For storing the step numbers that the agent is in suitable range
        self._stepsInRange = 0

        # Muscle activation
        self.bicepsActivation = 0
        self.tricepsActivation = 0

# PART 03- Gymnasium environmert (reward definitions) ======================================================
# ==========================================================================================================
        self.weights = rewardWeights
        
        # Last episode potential
        self.prevShapingReward = 0

        # DEBUG params
        self.accumulatedRewards = {
            "accumulatedDistanceReward": self.__distanceReward(self.state[0]) * self.weights["distance"],
            "accumulatedVelocityReward": self.__velocityReward(self.state[1]) * self.weights["velocity"],
            "accumulatedShapingReward": 0,
            "accumulatedStep": 0,
            "accumulatedInRangeReward": 0,
            "accumulatedInRangeReward_theta": 0,
            "accumulatedInRangeReward_thetaDot": 0,
            "accumulatedTerminationReward_success": 0,
            "accumulatedTerminationReward_failure": 0,
            "accumulatedRelativeShapingReward": 0,
            "accumulatedTruncationReward": 0,
            "accumulatedRewardSum": 0
        }
        self.stepWiseParams = {
            "distanceReward": self.__distanceReward(self.state[0]) * self.weights["distance"],
            "velocityReward": self.__velocityReward(self.state[1]) * self.weights["velocity"],
            "shapingReward": 0,
            "inRangeReward": 0,
            "inRangeReward_theta": 0,
            "inRangeReward_thetaDot": 0,
            "terminationReward_success": 0,
            "terminationReward_failure": 0,
            "relativeShapingReward": 0,
            "truncationReward": 0,
            "rewardSum": 0,
            "100WinRatio": 0,
            "overallWinRatio": 0,
            "stepNumber": 0
        }
        self.endHistory = [] # DEBUG - True for success, False for failure
        self.overallWinCount = 0 # DEBUG
        self.overallLossCount = 0 # DEBUG
        self.action_history = [0 for i in range(5)]  # DEBUG

    def __velocityReward(self, relativeThetaDot):
        """
        Args:
            relativeThetaDot (float): The relative thetaDot from target thetaDot 
        """
        err = relativeThetaDot / self._omegaRange[0 if relativeThetaDot < self.targetVelocity else 1] # Normalize Omega
        return np.power(err,2)

    def __distanceReward(self, relativeTheta):
        """
        Args:
            relativeTheta (float): The relative theta from target theta 
        """
        err = relativeTheta / self._thetaRange[0 if relativeTheta < self.target else 1] # Normalize theta
        return np.power(err,2)

    def __angular_error(self, current: np.float32, target: np.float32) -> np.float32:
        """
        Compute the shortest signed angular difference between two angles, 
        accounting for circular wrapping in the range [-π, π] radians. This
        function is critical for rotational systems (e.g., robotic arms, 
        pendulums) where angles are periodic and linear subtraction would 
        yield incorrect errors.

        Args:
        current (np.float32): Current angle in radians. Can be any real number (automatically wrapped).
        target (np.float32): Target angle in radians. Can be any real number (automatically wrapped).

        Returns:
            Signed angular error in radians, in the range [-π, π]. 

        * Uses `math.arctan2` for numerical stability and correct quadrant handling.
        * Equivalent to `(current - target + π) % (2π) - π` but more numerically robust.
        """
        return np.arctan2(np.sin(current - target), np.cos(current - target))

    def __checkInRange(self, theta, thetaDot):
        """
        Checks to see weather the agent is in the suitable range
        """
        # Check numpy closeness as well to avoid rounding errors
        cond1 = (self.envParams["suitableThetaRange"][0] <= theta <= self.envParams["suitableThetaRange"][1]) or np.isclose(self.envParams["suitableThetaRange"][0], theta) or np.isclose(self.envParams["suitableThetaRange"][1], thetaDot)
        cond2 = (self.envParams["suitableOmegaRange"][0] <= thetaDot <= self.envParams["suitableOmegaRange"][1]) or np.isclose(self.envParams["suitableOmegaRange"][0], thetaDot) or np.isclose(self.envParams["suitableOmegaRange"][1], thetaDot)
        
        if(cond1 and cond2):
            return True
        return False
    
    def getActualState(self, relativeState = None):
        """
        Returns the actual state of the agent.
        """

        if not relativeState:
            return self.state
        
        return (self.target + relativeState[0], self.targetVelocity + relativeState[1])

    def __calcRewards(self):
        # Calculate rewards
        __distreward = self.__distanceReward(self.state[0])
        __velReward = self.__velocityReward(self.state[1])
        __stepReward = 1
        __inRangeReward = 0
        __inRangeThetaReward = 0
        __inRangeThetaDotReward = 0
        __successReward = 0
        __failureReward = 0
        __truncationReward = 0

        reward = 0
        terminated = False
        truncated = False
        won = False

        # Suitability conditions
        cond1 = (self.envParams["suitableThetaRange"][0] <= self.state[0] <= self.envParams["suitableThetaRange"][1]) or np.isclose(self.envParams["suitableThetaRange"][0], self.state[0]) or np.isclose(self.envParams["suitableThetaRange"][1], self.state[0])
        cond2 = (self.envParams["suitableOmegaRange"][0] <= self.state[1] <= self.envParams["suitableOmegaRange"][1]) or np.isclose(self.envParams["suitableOmegaRange"][0], self.state[1]) or np.isclose(self.envParams["suitableOmegaRange"][1], self.state[1])

        if cond1:
            __inRangeThetaReward = 1

        if cond2:
            __inRangeThetaDotReward = 1

        if cond1 and cond2:
            self._stepsInRange += 1
            __inRangeReward = 1 # Add a small reward for being in suitable range in every step
            # print("IN RANGE")
            
            # Termination condition: Forearm position and velocity inside the range for 10 timesteps
            if 10 < self._stepsInRange:
                __successReward = 1
                terminated = True
                won = True
                print("REACHED SUCCESS")
        else:
            self._stepsInRange = 0

        # Termination condition: Forearm out of bounds
        if not (self.envParams["thetaMin"] < self.state[0] < self.envParams["thetaMax"]) or np.isclose(self.envParams["thetaMin"], self.state[0]) or np.isclose(self.envParams["thetaMax"], self.state[0]):
            __failureReward = -1
            terminated = True

        # Truncation condition: Surpassing maxStepNumber
        if self.envParams["maxStepNumber"] <= self._stepNum:
            __failureReward = -1
            truncated = True

        # DEBUG
        if __successReward == 1: 
            self.endHistory.append(True)
            self.overallWinCount += 1
        elif __failureReward == -1: 
            self.endHistory.append(False)
            self.overallLossCount += 1

        __distreward = self.weights["distance"] * __distreward
        __velReward = self.weights["velocity"] * __velReward
        __stepReward = self.weights["step"] * __stepReward
        __inRangeReward = self.weights["inRange"] * __inRangeReward
        __inRangeThetaReward = self.weights["inRangeTheta"] * __inRangeThetaReward
        __inRangeThetaDotReward = self.weights["inRangeThetaDot"] * __inRangeThetaDotReward
        __successReward = self.weights["termination_success"] * __successReward
        __failureReward = self.weights["termination_failure"] * __failureReward
        __truncationReward = self.weights["truncation"] * __truncationReward

        # Subtract the reward of previous step to avoid rewarding the same progress multiple times
        __relativeShapingReward = self.__shapingReward() * self.weights["shaping"] - self.prevShapingReward 
        reward = __relativeShapingReward + __stepReward + __inRangeReward + __inRangeThetaReward + __inRangeThetaDotReward + __successReward + __failureReward + __truncationReward
        # print("\nrelative shaping reward", __relativeShapingReward, "\nstep reward", __stepReward, "\nin range reward", __inRangeReward, "\nTheta in range reward", __inRangeThetaReward, "\nTheta dot in range reward", __inRangeThetaDotReward, "\nsccess reward", __successReward, "\nfailure reward", __failureReward, "\ntruncation reward", __truncationReward)
        # print("===================")

        # Update the shaping reward
        self.prevShapingReward = self.__shapingReward()

        # DEBUG
        self.accumulatedRewards["accumulatedDistanceReward"] += __distreward # DEBUG
        self.accumulatedRewards["accumulatedVelocityReward"] += __velReward # DEBUG
        self.accumulatedRewards["accumulatedShapingReward"] += self.__shapingReward() # DEBUG
        self.accumulatedRewards["accumulatedStep"] += __stepReward # DEBUG
        self.accumulatedRewards["accumulatedInRangeReward"] += __inRangeReward # DEBUG
        self.accumulatedRewards["accumulatedInRangeReward_theta"] += __inRangeThetaReward # DEBUG
        self.accumulatedRewards["accumulatedInRangeReward_thetaDot"] += __inRangeThetaDotReward # DEBUG
        self.accumulatedRewards["accumulatedTerminationReward_success"] += __successReward # DEBUG
        self.accumulatedRewards["accumulatedTerminationReward_failure"] += __failureReward # DEBUG
        self.accumulatedRewards["accumulatedTruncationReward"] += __truncationReward # DEBUG
        self.accumulatedRewards["accumulatedRelativeShapingReward"] += __relativeShapingReward # DEBUG
        self.accumulatedRewards["accumulatedRewardSum"] += reward # DEBUG
        self.accumulatedRewards["stepNumber"] = self._stepNum # DEBUG

        # DEBUG
        self.stepWiseParams["distanceReward"] = __distreward # DEBUG
        self.stepWiseParams["velocityReward"] = __velReward # DEBUG
        self.stepWiseParams["shapingReward"] = self.__shapingReward() # DEBUG
        self.stepWiseParams["inRangeReward"] = __inRangeReward # DEBUG
        self.stepWiseParams["inRangeReward_theta"] = __inRangeThetaReward # DEBUG
        self.stepWiseParams["inRangeReward_thetaDot"] = __inRangeThetaDotReward # DEBUG
        self.stepWiseParams["terminationReward_success"] = __successReward # DEBUG
        self.stepWiseParams["terminationReward_failure"] = __failureReward # DEBUG
        self.stepWiseParams["relativeShapingReward"] = __relativeShapingReward # DEBUG
        self.stepWiseParams["truncationReward"] = __truncationReward # DEBUG
        self.stepWiseParams["rewardSum"] = reward # DEBUG
        self.stepWiseParams["100WinRatio"] = self.endHistory.count(True) / (self.endHistory.count(True) + self.endHistory.count(False) + 1e-6) * 100 # DEBUG
        self.stepWiseParams["overallWinRatio"] = self.overallWinCount / (self.overallWinCount + self.overallLossCount + 1e-6) * 100 # DEBUG
        self.stepWiseParams["stepNumber"] = self._stepNum
        
        if terminated or truncated:
            self.envSave = self.accumulatedRewards

        return reward, terminated, truncated, won, [__distreward, __velReward, __inRangeReward, __inRangeThetaReward, __inRangeThetaDotReward, __stepReward, self.__shapingReward(), __relativeShapingReward]

    def __shapingReward(self):
        """
        Calculates the shaping reward of the current step. The higher 
        the reward, the more suitable the current state is. By default, 
        this number will be zero at best (When reached the target state)
        """
        # Both are negative numbers
        __distreward = self.__distanceReward(self.state[0]) * self.weights["distance"]
        __velReward = self.__velocityReward(self.state[1]) * self.weights["velocity"]

        return __distreward + __velReward

# PART 04- Gymnasium environmert (reset) ===================================================================
# ==========================================================================================================
    def reset(self, initState = None, seed = None, options = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed=seed)

        if initState is not None:
            # Use a pre-defined starting point.
            # Should have the same diemnsions as self.observation_space
            self.state = initState
        else:
            # Reset to a random state
            # self.state = np.array([rng.uniform(0, self.envParams["thetaMax"]), rng.uniform(-self.envParams["omegaMax"], self.envParams["omegaMax"])], dtype=np.float32)

            # If arm is already ina suitable range, make a new random state
            self.state = np.array([rng.uniform(0, self.envParams["thetaMax"]), 0], dtype=np.float32)
            while self.__checkInRange(self.state[0], self.state[1]):
                self.state = np.array([rng.uniform(0, self.envParams["thetaMax"]), 0], dtype=np.float32)

        # Restart the step number
        self._stepNum = 0
        self._stepsInRange = 0
        self._episode_reward = 0
        self.envSave = None

        # Restart muscle params
        self.bicepsActivation = 0
        self.tricepsActivation = 0

        # Restart the shaping reward
        self.prevShapingReward = 0

        # DEBUG params
        self.accumulatedRewards = {
            "accumulatedDistanceReward": self.__distanceReward(self.state[0]) * self.weights["distance"],
            "accumulatedVelocityReward": self.__velocityReward(self.state[1]) * self.weights["velocity"],
            "accumulatedShapingReward": 0,
            "accumulatedStep": 0,
            "accumulatedInRangeReward": 0,
            "accumulatedInRangeReward_theta": 0,
            "accumulatedInRangeReward_thetaDot": 0,
            "accumulatedTerminationReward_success": 0,
            "accumulatedTerminationReward_failure": 0,
            "accumulatedRelativeShapingReward": 0,
            "accumulatedTruncationReward": 0,
            "accumulatedRewardSum": 0
        }
        self.stepWiseParams = {
            "distanceReward": self.__distanceReward(self.state[0]) * self.weights["distance"],
            "velocityReward": self.__velocityReward(self.state[1]) * self.weights["velocity"],
            "shapingReward": 0,
            "inRangeReward": 0,
            "inRangeReward_theta": 0,
            "inRangeReward_thetaDot": 0,
            "terminationReward_success": 0,
            "terminationReward_failure": 0,
            "relativeShapingReward": 0,
            "truncationReward": 0,
            "rewardSum": 0,
            "100WinRatio": 0,
            "overallWinRatio": 0,
            "stepNumber": 0
        }
        self.action_history = [0 for i in range(5)]  # DEBUG

        # DEBUG
        if 200 < len(self.endHistory): 
            self.endHistory.pop(0)
        
        return np.array([self.state[0]-self.target, self.state[1]-self.targetVelocity]), {}

# PART 04- Gymnasium environmert (step) ====================================================================
# ==========================================================================================================
    def step(self, action):
        """
        Takes a step and solve the differential equation. The step duration
        is self.envParams["dt"] and is different from the timestep number which
        should be handled by the training loop.

        * Note that state is a list of [angle, angular velocity] of the elbow

        Args:
            action (int): 
                0 => Increase biceps activation a single unit
                1 => Decrease biceps activation a single unit
                2 => Increase triceps activation a single unit
                3 => Decrease triceps activation a single unit
                4 => Do nothing

        Returns:
            state, reward, terminated, truncated, info
        """
        if action == 0:
            self.bicepsActivation = np.clip(self.bicepsActivation + self.envParams["activationUnit"], 0, 1)
        elif action == 1:
            self.bicepsActivation = np.clip(self.bicepsActivation - self.envParams["activationUnit"], 0, 1)
        elif action == 2:
            self.tricepsActivation = np.clip(self.tricepsActivation + self.envParams["activationUnit"], 0, 1)
        elif action == 3:
            self.tricepsActivation = np.clip(self.tricepsActivation - self.envParams["activationUnit"], 0, 1)
        elif action == 4:
            pass # Do nothing

        terminated = False
        truncated  = False
        reward = 0

        # # Unpack action (two values in [0, 1]) - Commented out because not necessary
        # Calculate muscle forces
        _z = [self.bicepsActivation, self.tricepsActivation]
        
        bicepsForce, tricepsForce, _bicepsLength, _tricepsLength = self.calcMuscleForce(self.state[0], self.state[1], self.muscleParams, _z)
        
        # Torque calculations (simplified)
        _bicepsTorqueArm = MuscleParams.tendon_start_to_elbow * np.sin(self.state[0]) * MuscleParams.elbow_to_biceps_end / _bicepsLength
        _tricepsTorqueArm = MuscleParams.tendon_start_to_elbow * np.sin(self.state[0]) * MuscleParams.elbow_to_triceps_end / _tricepsLength
        
        torque = (bicepsForce * _bicepsTorqueArm) - (tricepsForce * _tricepsTorqueArm) 

        # Differential equations for rotational motion
        alpha = torque / self.envParams["forearmInertia"] # moment of inertia of 0.02 kg*m^2

        # Impose boundary rules for angular velocity (omega)
        if self.envParams["thetaMin"] < self.state[0]  and self.state[0] < self.envParams["thetaMax"]:
            self.state[1] = np.clip(self.state[1] + alpha * self.envParams["dt"], -self.envParams["omegaMax"], self.envParams["omegaMax"])
        elif self.state[0] < self.envParams["thetaMax"] + .1 and alpha < 0: 
            # Added for environment correctness. It will never get here, because agent terminates at thetaMax
            self.state[1] = np.clip(alpha * self.envParams["dt"], -self.envParams["omegaMax"], 0)
        elif self.state[0] > self.envParams["thetaMin"] - .1 and alpha > 0:
            # Added for environment correctness. It will never get here, because agent terminates at thetaMax
            self.state[1] = np.clip(alpha * self.envParams["dt"], 0, self.envParams["omegaMax"])
        else:
            self.state[1] = 0

        # Update angle
        self.state[0] += self.state[1] * self.envParams["dt"]

        # Impose boundary rules for angle (theta)
        self.state[0] = np.clip(self.state[0], self.envParams["thetaMin"], self.envParams["thetaMax"])

        # Get relative states
        self.relativeState = np.array([self.state[0] - self.target,  self.state[1]- self.targetVelocity])

        # Increase step number
        self._stepNum += 1

        # Get rewards
        reward, terminated, truncated, won, lstRewards = self.__calcRewards()
        
        info = {
            "distanceReward": lstRewards[0], 
            "velocityReward": lstRewards[1], 
            "inRangeReward": lstRewards[2], 
            "inRangeThetaReward": lstRewards[3], 
            "inRangeThetaDotReward": lstRewards[4], 
            "stepReward": lstRewards[5], 
            "shapingReward": lstRewards[6],
            "relativeShapingReward": lstRewards[7],
            "stepNum": self._stepNum,
            "bicepsForce": bicepsForce,
            "tricepsForce": tricepsForce,
            "bicepsTorque": _bicepsTorqueArm * bicepsForce,
            "tricepsTorque": _tricepsTorqueArm * tricepsForce,
            "torque": torque,
            "isWon": won
            } # DEBUG
        self.action_history[action] = self.action_history[action] + 1  # DEBUG

        return np.array([self.state[0]-self.target, self.state[1]-self.targetVelocity]), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # PART 05- Hill's model for Muscle Force cal. ==============================================================
    # 5-1 ======================================================================================================
    def hillForceLength_active(self, L, L_min = None, L_0 = 1, L_max = None):
        """
        Calculate the normalized active force-length relationship.
        
        Parameters:
            L (array): Muscle length (normalized)
            L_min (float): Minimum length where active force is zero (e.g., 0.6 L_0)
            L_0 (float): Optimal length where maximum force is produced (e.g., 1.0 L_0)
            L_max (float): Maximum length where active force is zero (e.g., 1.6 L_0)
        
        Returns:
            fActive (array): Normalized active force (0 to 1)
        """
        fActive = np.zeros_like(L)

        # # Method 1 (Not vectorizable)
        # # Piecewise linear force-length relationship
        # for i, length in enumerate(L):
        #     if length <= L_min:
        #         fActive[i] = 0.0
        #     elif L_min < length < .71:
        #         fActive[i] = (.73-0)/(0.71-L_min) * length - (.73-0)/(0.71-L_min) * L_min
        #     elif .71 <= length < L_0:
        #         fActive[i] = (1-.73)/(1-.71) * length + 0.069
        #     elif L_0 <= length < 1.09:
        #         fActive[i] = 1
        #     elif 1.09 <= length < L_max:
        #         fActive[i] = -1 / (L_max - 1.09) * length + (1 / (L_max - 1.09)) * L_max
        #     else:
        #         fActive[i] = 0.0
        
        # # Method 2 (Vectorizable)
        # fActive = np.exp(-((L - L_0) ** 2) / 0.1)

        # Method 3 (Vectorizable)
        # using fourier expansion
        mask = (L_min < L) & (L < L_max)
        a0 = 0.5511
        w = 4.4301
        A = np.array([-0.0430, 0.0599, -0.0458, 0.0199, -0.0058])
        B = np.array([-0.4770, 0.0723, 0.0148, -0.0096, 0.0013])
        C = lambda x: np.array([np.cos(x), np.cos(2*x), np.cos(3*x), np.cos(4*x), np.cos(5*x)])
        S = lambda x: np.array([np.sin(x), np.sin(2*x), np.sin(3*x), np.sin(4*x), np.sin(5*x)])
        ff = lambda L: a0 + np.dot(A, C(w*L)) + np.dot(B, S(w*L)) # Explansion
        _force = ff(L[mask])
        fActive[mask] = _force
        
        return fActive

    # 5-2 ======================================================================================================
    def hillForceVelocity(self, v, F_max=1.0, a=0.25, b=0.25, eccentric_scale=1.4):
        """
        Compute muscle force based on Hill's force-velocity relationship.
        
        Parameters:
        - v: Muscle contraction velocity (normalized to maximum shortening velocity, v_max).
            Positive = shortening, Negative = lengthening.
        - F_max: Maximum isometric force (at v=0).
        - a, b: Hill's constants (dimensionless, typically a/F_max ≈ 0.25).
        - eccentric_scale: Scaling factor for eccentric (lengthening) force enhancement.
        
        Returns:
        - F: Normalized muscle force (F/F_max).
        """
        F = np.zeros_like(v)
        
        # Concentric (shortening) region: v > 0
        shortening = v > 0
        F[shortening] = (F_max * b - a * v[shortening]) / (b + v[shortening])
        
        # Eccentric (lengthening) region: v < 0
        lengthening = v <= 0
        F[lengthening] = (eccentric_scale * F_max - (eccentric_scale * F_max - F_max) * np.exp(-abs(v[lengthening]) / b))
        
        return F

    # 5-3 ======================================================================================================
    def calcMuscleForce(self, theta: float, omega: float, params: MuscleParams, EMG: List[float]) -> Tuple[float, float]:
        """
        Compute muscle forces based on joint angle and velocity
        
        Args:
            theta: Joint angle (radians)
            omega: Angular velocity (rad/s)
            params: Muscle parameters
            EMG: Muscle activation levels [biceps, triceps]
        
        Returns:
            Biceps force, Triceps force, Biceps length, Triceps length
        """
        # Muscle-tendon unit lengths
        _tendonStartToElbow = params.tendon_start_to_elbow
        _bicepsTendonSlackLength = params.tendon_slack_length_Biceps
        _tricepsTendonSlackLength = params.tendon_slack_length_Triceps
        _elbowToMuscleEnd_Biceps = params.elbow_to_biceps_end
        _elbowToMuscleEnd_Triceps = params.elbow_to_triceps_end

        L_biceps = np.sqrt(np.power(_tendonStartToElbow,2) + np.power(_elbowToMuscleEnd_Biceps,2) - 2 * _elbowToMuscleEnd_Biceps * _tendonStartToElbow * np.cos(np.pi - theta)) - _bicepsTendonSlackLength
        L_triceps = np.sqrt(np.power(_tendonStartToElbow,2) + np.power(_elbowToMuscleEnd_Triceps,2) - 2 * _elbowToMuscleEnd_Triceps * _tendonStartToElbow * np.cos(theta)) - _tricepsTendonSlackLength


        V_biceps = (-MuscleParams.elbow_to_biceps_end * omega * np.sin(theta) * _elbowToMuscleEnd_Biceps / L_biceps)  / params.max_contraction_velocity_Biceps
        V_triceps = (MuscleParams.elbow_to_triceps_end * omega * np.sin(theta) * _elbowToMuscleEnd_Triceps / L_triceps) / params.max_contraction_velocity_Triceps

        hill_L_biceps = L_biceps / params.optimal_fiber_length_Biceps
        hill_L_triceps = L_triceps / params.optimal_fiber_length_Triceps

        V_biceps = V_biceps / params.max_contraction_velocity_Biceps
        V_triceps = V_triceps / params.max_contraction_velocity_Triceps

        # Hill's model for muscle force
        F_biceps = EMG[0] * self.hillModelForce(hill_L_biceps, V_biceps, params, "Biceps")
        F_triceps = EMG[1] * self.hillModelForce(hill_L_triceps, V_triceps, params, "Triceps")
        
        return F_biceps, F_triceps, L_biceps, L_triceps

    # 5-4 ======================================================================================================
    def hillModelForce(self, L: float, V: np.ndarray, params: MuscleParams, name: str) -> np.ndarray:
        """
        Hill's muscle model for force generation
        
        Args:
            L: Muscle length
            V: Muscle velocity (can be array)
            params: Muscle parameters
            name: Muscle name (Biceps or Triceps)
        
        Returns:
            Force generated by muscle's length change
            Force generated by muscle's velocity
            Total force, multiplied by maximum isometric force
        """
        # Handle scalar or array input
        V = np.atleast_1d(V)
        
        F_length = self.hillForceLength_active(
            L, 
            L_min = params.Length_min,
            L_max = params.Length_max
        )
        F_velocity = self.hillForceVelocity(V)
        
        return F_length * F_velocity * getattr(params, f"max_isometric_force_{name}")


    # Visualizing the simulation
    def arm_animation(self, time: np.ndarray, theta_results: np.ndarray, omega_results: np.ndarray, F_biceps_results: np.ndarray, F_triceps_results: np.ndarray):
        """
        Create animated visualization of arm movement
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Arm visualization
        ax1.set_xlim(-0.4, 0.4)
        ax1.set_ylim(-0.1, 0.5)
        ax1.set_aspect('equal')
        ax1.set_title('Arm Animation')
        ax1.grid(True, alpha=0.3)
        
        # Create arm segments
        upper_arm_length = 0.3
        forearm_length = 0.25
        
        line_upper, = ax1.plot([], [], 'b-', linewidth=8, label='Upper arm')
        line_forearm, = ax1.plot([], [], 'r-', linewidth=6, label='Forearm')
        point_elbow, = ax1.plot([], [], 'ko', markersize=8, label='Elbow')
        point_wrist, = ax1.plot([], [], 'go', markersize=6, label='Wrist')
        
        ax1.legend()
        
        # Force visualization
        ax2.set_xlim(0, len(time))
        ax2.set_ylim(0, max(max(F_biceps_results), max(F_triceps_results)) * 1.1)
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Muscle Forces')
        ax2.grid(True, alpha=0.3)
        
        line_biceps, = ax2.plot([], [], 'b-', linewidth=2, label='Biceps')
        line_triceps, = ax2.plot([], [], 'r-', linewidth=2, label='Triceps')
        ax2.legend()
        
        def animate(frame):
            if frame < len(theta_results):
                theta = theta_results[frame]
                
                # Calculate positions
                elbow_x, elbow_y = 0, 0
                wrist_x = forearm_length * np.cos(theta)
                wrist_y = forearm_length * np.sin(theta)
                shoulder_x = -upper_arm_length * np.cos(np.pi/2)
                shoulder_y = -upper_arm_length * np.sin(np.pi/2)
                
                # Update arm segments
                line_upper.set_data([shoulder_x, elbow_x], [shoulder_y, elbow_y])
                line_forearm.set_data([elbow_x, wrist_x], [elbow_y, wrist_y])
                point_elbow.set_data([elbow_x], [elbow_y])
                point_wrist.set_data([wrist_x], [wrist_y])
                
                # Update force plot
                line_biceps.set_data(range(frame + 1), F_biceps_results[:frame + 1])
                line_triceps.set_data(range(frame + 1), F_triceps_results[:frame + 1])
            
            return line_upper, line_forearm, point_elbow, point_wrist, line_biceps, line_triceps
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(time), interval=100, blit=True, repeat=True)
        plt.tight_layout()
        return anim


    def createArmAnimation(self, time, theta_results, omega_results, target, save_filename=None, theta_reward=None, omega_reward=None, data_df=None):
        """
        Create animated visualization of arm movement.

        Optionally, plot extra time-series data (one subplot per DataFrame column)
        stacked vertically on the right side of the main animation. All right-side
        subplots share the same x-axis (time) and include a moving vertical line
        indicating the current animation time.

        Args:
            time : array-like
                Time array for animation
            theta_results : array-like
                Angle results for each time step
            omega_results : array-like
                Angular velocity results for each time step
            target : float
                Target angle
            save_filename : str, optional
                Filename to save animation (e.g., 'arm_animation.mp4')
            theta_reward : array-like, optional
                Theta reward values for each time step
            omega_reward : array-like, optional
                Omega reward values for each time step
            data_df : pandas.DataFrame, optional
                Each column is plotted as a separate right-side subplot.
        """
        # Get necessary constants
        params = MuscleParams()

        deg = np.pi / 180

        show_series = isinstance(data_df, pd.DataFrame) and len(data_df.columns) > 0
        show_rewards = (not show_series) and theta_reward is not None and omega_reward is not None

        axes_right = []
        progress_lines = []

        # Set up the figure and subplots
        if show_series:
            n_series = len(data_df.columns)
            fig = plt.figure(figsize=(15, max(6, 1.6 * n_series)))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.0], wspace=0.35)
            ax1 = fig.add_subplot(gs[0, 0])
            gs_right = gs[0, 1].subgridspec(int(np.ceil(n_series/3)),3, hspace=0.5)
        elif show_rewards:
            fig = plt.figure(figsize=(8, 10))
            # Use gridspec to control subplot sizes
            # Height ratios: main plot gets 3 units, each reward plot gets 1 unit
            gs = fig.add_gridspec(4, 1, height_ratios=[5, 1, 1, 1], hspace=0.4)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[3, 0])
        else:
            fig = plt.figure(figsize=(6, 6))
            ax1 = plt.subplot(1, 1, 1)
        
        # Configure main animation subplot
        ax1.set_xlim(-.5, .5)
        ax1.set_ylim(-.6, 0.1)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Arm Animation')
        
        # Initialize empty line objects for animation
        upper_arm_line, = ax1.plot([], [], 'b-', linewidth=2, label='Upper arm')
        forearm_line, = ax1.plot([], [], 'r-', linewidth=3, label='Forearm')
        bicepsLine, = ax1.plot([], [], 'k:', linewidth=1, label='Biceps')
        tricepsLine, = ax1.plot([], [], 'k:', linewidth=1, label='Triceps')
        elbow_point, = ax1.plot([], [], 'ko', markersize=4, label='Elbow')
        wrist_point, = ax1.plot([], [], 'go', markersize=3, label='Wrist')
        target_point, = ax1.plot([], [], 'ro', markersize=1, label='Target')
        target_line_up, = ax1.plot([], [], ls = '--', color = 'r', linewidth=.7, label='Target offset')
        target_line_down, = ax1.plot([], [], ls = '--', color = 'r', linewidth=.7, label='Target offset')
        ax1.legend()
        
        # Configure right-side DataFrame subplots if provided
        if show_series:
            for i, col in enumerate(list(data_df.columns)):
                ax = fig.add_subplot(gs_right[int(i/3), i%3], sharex=axes_right[0] if axes_right else None)

                y = data_df[col].to_numpy()
                ax.plot(time, y, linewidth=1, alpha=0.7)
                ax.set_xlim(time[0], time[-1])

                y_min = np.nanmin(y)
                y_max = np.nanmax(y)

                if (str(col)) == "theta":
                    ax.hlines([self.envParams["suitableThetaRange"][0], self.envParams["suitableThetaRange"][1]], np.min(time), np.max(time), colors="red", )
                    y_min = np.min([y_min, self.envParams["suitableThetaRange"][0]])
                    y_max = np.max([y_max, self.envParams["suitableThetaRange"][1]])
                elif (str(col)) == "thetaDot":
                    ax.hlines([self.envParams["suitableOmegaRange"][0], self.envParams["suitableOmegaRange"][1]], np.min(time), np.max(time), colors="red", )
                    y_min = np.min([y_min, self.envParams["suitableOmegaRange"][0]])
                    y_max = np.max([y_max, self.envParams["suitableOmegaRange"][1]])

                if not np.isfinite(y_min) or not np.isfinite(y_max):
                    y_min, y_max = -1.0, 1.0
                if y_max == y_min:
                    pad = max(1e-6, abs(y_max) * 0.1)
                else:
                    pad = 0.1 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)

                ax.set_title(str(col))
                ax.grid(True, alpha=0.3)

                vline, = ax.plot([], [], 'r-', linewidth=1)
                axes_right.append(ax)
                progress_lines.append(vline)

            if axes_right:
                axes_right[-1].set_xlabel('Time (s)')
                for ax in axes_right[:-1]:
                    ax.tick_params(labelbottom=False)

        # Configure reward subplots if needed
        if show_rewards:
            # Theta reward subplot
            ax2.plot(time, theta_reward, 'b-', linewidth=1, alpha=0.7)
            ax2.set_xlim(time[0], time[-1])
            ax2.set_ylim(min(theta_reward) * 1.1, max(theta_reward) * 1.1)
            ax2.set_ylabel('Theta Reward')
            ax2.set_title('Theta Reward Progress')
            ax2.grid(True, alpha=0.3)
            theta_progress_line, = ax2.plot([], [], 'r-', linewidth=2)
            
            # Omega reward subplot
            ax3.plot(time, omega_reward, 'g-', linewidth=1, alpha=0.7)
            ax3.set_xlim(time[0], time[-1])
            ax3.set_ylim(min(omega_reward) * 1.1, max(omega_reward) * 1.1)
            ax3.set_ylabel('Omega Reward')
            ax3.set_title('Omega Reward Progress')
            ax3.grid(True, alpha=0.3)
            omega_progress_line, = ax3.plot([], [], 'r-', linewidth=2)
            
            # Omega subplot
            ax4.plot(time, omega_results, 'g-', linewidth=1, alpha=0.7)
            ax4.set_xlim(time[0], time[-1])
            ax4.set_ylim(min(omega_results) * 1.1, max(omega_results) * 1.1)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Omega')
            ax4.set_title('Omega Progress')
            ax4.grid(True, alpha=0.3)
            thetaDot_progress_line, = ax4.plot([], [], 'r-', linewidth=2)
        
        # Animation function
        def animate(frame):
            # Get current angle and time
            current_theta = theta_results[frame]
            current_time = time[frame]
            
            # Joint positions
            shoulderLoc = (0,0)
            elbowLoc = (0, - params.upper_arm_length)
            tendonStartLoc = (0, - params.upper_arm_length + params.tendon_start_to_elbow)

            wristLoc = (0 + params.forearm_length * np.sin(current_theta), - params.upper_arm_length - params.forearm_length * np.cos(current_theta))
            bicepsEndLoc = (0 + params.elbow_to_biceps_end * np.sin(current_theta), - params.upper_arm_length - params.elbow_to_biceps_end * np.cos(current_theta))
            tricepsEndLoc = (0 - params.elbow_to_triceps_end * np.sin(current_theta), - params.upper_arm_length + params.elbow_to_triceps_end * np.cos(current_theta))
            targetLoc = (0 + params.forearm_length * np.sin(target), - params.upper_arm_length - params.forearm_length * np.cos(target))

            # Lines for main animation
            upper_arm_line.set_data([shoulderLoc[0], elbowLoc[0]], [shoulderLoc[1], elbowLoc[1]])
            forearm_line.set_data([tricepsEndLoc[0], wristLoc[0]], [tricepsEndLoc[1], wristLoc[1]])
            bicepsLine.set_data([tendonStartLoc[0], bicepsEndLoc[0]], [tendonStartLoc[1], bicepsEndLoc[1]])
            tricepsLine.set_data([tendonStartLoc[0], tricepsEndLoc[0]], [tendonStartLoc[1], tricepsEndLoc[1]])

            if 1 < len(target):
                upper_target_point = (0 + params.forearm_length * np.sin(target[1]), - params.upper_arm_length - params.forearm_length * np.cos(target[1]))
                lower_target_point = (0 + params.forearm_length * np.sin(target[2]), - params.upper_arm_length - params.forearm_length * np.cos(target[2]))
                target_line_up.set_data([elbowLoc[0], upper_target_point[0]], [elbowLoc[1], upper_target_point[1]])
                target_line_down.set_data([elbowLoc[0], lower_target_point[0]], [elbowLoc[1], lower_target_point[1]])
            
            # Points for main animation
            elbow_point.set_data([elbowLoc[0]], [elbowLoc[1]])
            wrist_point.set_data([wristLoc[0]], [wristLoc[1]])
            target_point.set_data([targetLoc[0]], [targetLoc[1]])
            
            # Update title with current angle
            ax1.set_title(f'Arm Animation - Angle: {current_theta/deg:.1f}°, Time: {time[frame]:.2f}s')
            
            # Update reward progress lines if rewards are provided
            animated_objects = [upper_arm_line, forearm_line, bicepsLine, tricepsLine, target_line_up, target_line_down, elbow_point, wrist_point, target_point]
            
            if show_series:
                for ax, vline in zip(axes_right, progress_lines):
                    y_range = ax.get_ylim()
                    vline.set_data([current_time, current_time], y_range)
                animated_objects.extend(progress_lines)

            if show_rewards:
                # Update vertical progress lines
                theta_y_range = ax2.get_ylim()
                omega_y_range = ax3.get_ylim()
                theaDot_y_range = ax4.get_ylim()
                
                theta_progress_line.set_data([current_time, current_time], theta_y_range)
                omega_progress_line.set_data([current_time, current_time], omega_y_range)
                thetaDot_progress_line.set_data([current_time, current_time], theaDot_y_range)
                
                animated_objects.extend([theta_progress_line, omega_progress_line, thetaDot_progress_line])
            
            return animated_objects
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(time), interval=5, blit=True, repeat=True)
        
        # Save animation if filename is provided
        if save_filename:
            print(f"Saving animation to {save_filename}...")
            # You can adjust the writer and parameters as needed
            # For MP4: writer='ffmpeg'
            # For GIF: writer='pillow'
            anim.save(save_filename, writer='ffmpeg', fps=30, bitrate=1800)
            print(f"Animation saved successfully!")
        
        plt.tight_layout()
        plt.show()
        return fig, anim

    def plotProgress(self, hist, saveLoc):
        """
        Plots the training progress

        Args:
            hist (array): An array of dictionary containing the training history
            saveLoc (string): The directory to save the plot
        """
        # Prepare the data
        histDf = pd.DataFrame(hist)
        labelNames = [
            "distance reward", 
            "velocity reward", 
            "shaping reward", 
            "step reward", 
            "inRange reward",
            "inRange theta reward",
            "inRange thetaDot reward", 
            "success reward", 
            "failure reward", 
            "truncation reward", 
            "relative shaping reward",
            "sum",
            "step count"
        ]
        keys = [
            "accumulatedDistanceReward",
            "accumulatedVelocityReward",
            "accumulatedShapingReward",
            "accumulatedStep",
            "accumulatedInRangeReward",
            "accumulatedInRangeReward_theta",
            "accumulatedInRangeReward_thetaDot",
            "accumulatedTerminationReward_success",
            "accumulatedTerminationReward_failure",
            "accumulatedTruncationReward",
            "accumulatedRelativeShapingReward",
            "accumulatedRewardSum",
            "stepNumber"
        ]

        # Plot the performance
        # Create a figure with 7 subplots arranged vertically
        fig, axs = plt.subplots(len(keys), 1, figsize=(8, 20), sharex=True)

        # Plot each list in its own subplot
        for i in range(len(keys)):
            axs[i].scatter(histDf.index, histDf[keys[i]], 
                        s=2,                  # small marker size for a light appearance
                        color="black", 
                        alpha=1, 
                        marker='o',           # explicit circle marker (default, but clear)
                        linewidth=0)          # no edge lines on markers if unwanted
            axs[i].set_ylabel(f'{labelNames[i]}')
            axs[i].set_title(f'{labelNames[i]}')

        # Set common x-label
        axs[-1].set_xlabel('Index')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(os.path.join(saveLoc,"hill2dArm_training.png"))

        # Close all figures
        plt.close('all')
