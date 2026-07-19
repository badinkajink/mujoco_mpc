# Humanoid Bench in MuJoCo-MPC

This directory contains the re-implementation of some of the humanoid benchmark in MuJoCo-MPC. The original implementation is in [this repository](https://github.com/carlosferrazza/humanoid-bench).

## Reward to Residuals
MuJoCo-MPC uses residuals with multiple dimensions instead of a single reward.
The residuals should be 'close to zero' to indicate a good performance. So in each task, the first step is to compute the reward the same way it is done in the original implementation. 
Then, the first dimension of the residual is set to x - reward, where x is the maximum reward that can be achieved in the task.

## Additional Residuals
In addition to the reward residual, we also add additional residuals. We found them to be helpful to solve the task. 
To get the 'vanilla' version of the task, you can set the additional residuals weights to zero, using the sliders in the GUI.

## Robots
In the original implementation, they use a position controlled H1 robot from unitree, with per-task variants ('normal', a stronger version, and versions with various hands attached).

In this fork the primary robot is the position controlled Unitree H1-2 (the `*_H12` task variants: Walk, Push, Avoid, Lean, Stabilize, Upper). The original H1 is still registered for the Walk, Stand and Push tasks. A hands-equipped variant exists only where hands help (Lean H12 Hands); there are no 'stronger' variants here.

G1 (torque controlled) task classes and scene XMLs exist in the source (`Stand_G1`, `Walk_G1`, `G1_push`) but are not registered in `tasks.cc` and the build does not generate their base model assets, so they cannot be selected or run in this fork.

## Punch Task (not ported to this fork)
This task is not part of the original benchmark and has not been ported to this fork — no punch task exists here. (Upstream description: the robot has to punch a sphere target placed at a random position in front of it, with alternating hands.)
