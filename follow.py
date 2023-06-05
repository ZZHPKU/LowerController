import pybullet as p
import numpy as np
import pickle
from utilities import readLogFile

import time

with open(r'data/right_bin_to_left_bin_pipeline.pkl', "rb") as f:
    all_sols = pickle.load(f)

p.connect(p.GUI)
my_manipulator = p.loadURDF(r"model/iiwa14_my.urdf")
num_of_joints = p.getNumJoints(my_manipulator)

joint_array = []
jointLowerLimit_array = []
jointUpperLimit_array = []

for i in range(num_of_joints):
    a = p.getJointInfo(my_manipulator, i)
    if a[2] != 4:
        joint_array.append(i)
        jointLowerLimit_array.append(a[8])
        jointUpperLimit_array.append(a[9])

num_of_joints_work = len(joint_array)
desiredPosId = []

full_traj = False

# traj_origin = np.array(all_sols[2])
traj_origin = all_sols[0]
N_origin = traj_origin.shape[0]
t = np.linspace(0, 1.5, N_origin)
N = N_origin
t_line_intrep = np.linspace(0, 1.5, N)

traj = np.zeros(traj_origin.shape)
for j in range(traj_origin.shape[1]):
    traj[:, j] = np.interp(t_line_intrep, t, traj_origin[:, j])

time_step = 1.5 / (N - 1)
sub_step_num = 3
maxForce = 1000
# kp = 1.5/49*3.33
kp = 0.1
kd = 0.5

# result initialize
control = np.zeros((N - 1, num_of_joints_work))
if full_traj:
    control_traj = np.zeros((N, num_of_joints_work * 2))
else:
    control_traj = np.zeros((N, num_of_joints_work))

p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSubSteps=sub_step_num)
p.setGravity(0, 0, -10.)

# initialize
if full_traj:
    desiredPos = traj[0, :num_of_joints_work]
    desiredVel = traj[0, num_of_joints_work:]
else:
    desiredPos = traj[0]
    desiredVel = np.zeros((num_of_joints_work,))

for j in range(num_of_joints_work):
    p.setJointMotorControl2(my_manipulator,
                            joint_array[j],
                            p.POSITION_CONTROL,
                            targetPosition=desiredPos[j],
                            targetVelocity=desiredVel[j],
                            positionGain=kp,
                            velocityGain=kd,
                            force=maxForce)

for k in range(100):
    p.stepSimulation()
for j in range(num_of_joints_work):
    [jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque] = p.getJointState(my_manipulator,
                                                                                                   joint_array[j])
    if full_traj:
        control_traj[0, j] = jointPosition
        control_traj[0, num_of_joints_work + j] = jointVelocity
    else:
        control_traj[0, j] = jointPosition
time.sleep(0.5)

logID = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, r"mytest\\log.txt", maxLogDof=7,
                            logFlags=p.STATE_LOG_JOINT_TORQUES)

for k in range(1, N):
    if full_traj:
        desiredPos = traj[k, :num_of_joints_work]
        desiredVel = traj[k, num_of_joints_work:]
    else:
        desiredPos = traj[k]
        if k == N - 1:
            targetVelocity = np.zeros((num_of_joints_work,))
        else:
            targetVelocity = (traj[k + 1] - traj[k]) / time_step
    for j in range(num_of_joints_work):
        p.setJointMotorControl2(my_manipulator, joint_array[j],
                                controlMode=p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(my_manipulator,
                                joint_array[j],
                                p.POSITION_CONTROL,
                                targetPosition=desiredPos[j],
                                targetVelocity=desiredVel[j],
                                positionGain=kp,
                                velocityGain=kd,
                                force=maxForce)
    p.stepSimulation()
    for j in range(num_of_joints_work):
        [jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque] = p.getJointState(
            my_manipulator, joint_array[j])
        if full_traj:
            control_traj[k, j] = jointPosition
            control_traj[k, num_of_joints_work + j] = jointVelocity
        else:
            control_traj[k, j] = jointPosition
        control[k - 1, j] = appliedJointMotorTorque

    time.sleep(time_step)

p.stopStateLogging(logID)
p.disconnect()

a = np.array(readLogFile(r"data/log.txt", True))

traj_detail = a[:, 17:24]
control_detail = a[:, 31:38]

order = np.arange(N - 1)

print("error of the two log method:",
      np.linalg.norm(control - control_detail[sub_step_num * order + sub_step_num - 1] / sub_step_num))
print("trajectory follow error:", np.linalg.norm(control_traj - traj) ** 2 * time_step)
print("torque cost:", np.sum((control_detail / sub_step_num) ** 2 * time_step / sub_step_num))
