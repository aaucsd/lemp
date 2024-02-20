#!/usr/bin/env python


import warnings
warnings.filterwarnings("ignore")


import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.msg._RobotTrajectory import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Duration


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from math import pi, cos, dist, fabs
import numpy as np

from objects.static.voxel import VoxelObject

from environment.static.kuka_env import KukaEnv
from wrappers.obstacles import ObstaclePositionWrapper
from planner.learned.GNN_static_planner import GNNStaticPlanner
from objects.trajectory import WaypointLinearTrajectory
import pybullet as p
import pybullet_data
# import numpy as np 
# from math import pi

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True



class kuka_moveit(object):
    """kuka_moveit"""

    def __init__(self):
        super(kuka_moveit, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)
        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()

        group_name = "kuka_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        planning_frame = move_group.get_planning_frame()

        eef_link = move_group.get_end_effector_link()

        group_names = robot.get_group_names()


        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self, num):

        move_group = self.move_group
        current_joints = move_group.get_current_joint_values()

        tau = 2*pi
        if num == 1:
            tau = pi/2
        else:
            tau = 2*pi
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = pi/3
        joint_goal[1] = -tau / 8
        joint_goal[2] = pi/3
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 6  # 1/6 of a turn
        joint_goal[6] = pi/3

        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()


        # For testing:
        current_joints = move_group.get_current_joint_values()
        # print("The current joint states are : {}".format(current_joints))
        return all_close(joint_goal, current_joints, 0.01)



    def plan_cartesian_path(self, scale=1):

        move_group = self.move_group

        ##
        waypoints = []

        wpose = move_group.get_current_pose().pose
        wpose.position.z -= scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0 )  
        print("The datatype of the plan is {}".format(plan._type))
        return plan, fraction

        ## END_SUB_TUTORIAL

    def display_trajectory(self, plan):

        robot = self.robot
        # print(2)
        display_trajectory_publisher = self.display_trajectory_publisher
        # print(3)

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        # print(5)
        display_trajectory.trajectory_start = robot.get_current_state()
        # print(6)
        display_trajectory.trajectory.append(plan)
        # print(7)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)


    def execute_plan(self, plan):

        move_group = self.move_group

        move_group.execute(plan, wait=True)


# def linear(start,goal,t, t_max):
#     return (goal-start)*(t/t_max) + start
# def linear(waypoints, t, t_max):
#     i = ((t*len(waypoints)-1)/t_max)//1




def main():

    try:
        # print(1)

        kuka = kuka_moveit()
        env = KukaEnv(objects=[VoxelObject(base_orientation=[0, 0, 0, 1], base_position=[0, 1, 1], half_extents=[0.2, 0.2, 0.2])])
        env = ObstaclePositionWrapper(env)
        env.load()
        print("\n")
        
        # hand_limits = np.zeros(4)
        num_real_joints = 7
        # env.robot.limits_low[7:] = hand_limits.tolist()
        # env.robot.limits_high[7:] = hand_limits.tolist()
        # start = np.concatenate((np.array(kuka.move_group.get_current_joint_values()),hand_limits))
        # goal = np.concatenate((np.array([pi/3, -pi/16, pi/3,-pi/8,0,pi/12,pi/3]),hand_limits))

        while True:
            _, goal = env.robot.sample_random_init_goal()
            start = np.array(kuka.move_group.get_current_joint_values())
            if not env.edge_fp(start,goal) or True:
                result = GNNStaticPlanner(num_batch=100, model_args=dict(config_size=env.robot.config_dim, 
                                                                                embed_size=64, 
                                                                                obs_size=6)).plan(env, start, goal, timeout=('time', 100))
                if result.solution:
                    break


        while True:
            # if not env.edge_fp(start,goal) or True:
            result = GNNStaticPlanner(num_batch=100, model_args=dict(config_size=env.robot.config_dim, 
                                                                            embed_size=64, 
                                                                            obs_size=6)).plan(env, start, goal, timeout=('time', 100))
            if result.solution:
                break
        # print(True if result.solution else False)
        # print(2)
        # kuka.go_to_joint_state(joint_start_state)
        # print(8)
        # kuka.go_to_joint_state(2)

        # print(3)
        # cartesian_plan, fraction = kuka.plan_cartesian_path()
        # print(cartesian_plan.joint_trajectory)
        # print(rospy.Duration(2))
        # t = Duration()
        # t.data.secs = 0.1
        # print(t)
        # t_max = 10

        # kuka = kuka_moveit()
        traj = WaypointLinearTrajectory(result.solution)

        plan = RobotTrajectory()
        plan.joint_trajectory = JointTrajectory()
        plan.joint_trajectory.joint_names = ['lbr_iiwa_joint_1','lbr_iiwa_joint_2','lbr_iiwa_joint_3','lbr_iiwa_joint_4','lbr_iiwa_joint_5','lbr_iiwa_joint_6','lbr_iiwa_joint_7']
        # steps = 100 * (len(result.solution)-1)
        # t_step = 0.1
        # t_max = (steps-1)*t_step
        for timestep in np.linspace(0, len(traj.waypoints)-1, 100):
            point = JointTrajectoryPoint()
            joints = traj.get_spec(timestep)
            point.positions = joints[0:num_real_joints].tolist()
            env.robot.set_config(joints)
            p.performCollisionDetection()

            point.time_from_start.secs = int(timestep//1)
            point.time_from_start.nsecs = int((timestep % 1)*1e9 )
            plan.joint_trajectory.points.append(point)



        # for step in range(0,steps):
        #     t = step*t_step
            # joints = traj.get_spec(t).tolist()
            # joints = linear(waypoints, t,t_max)
            # point = JointTrajectoryPoint()
            # point.positions = joints.tolist()
            # time_msg = Duration()
            # time_msg.data.secs = t
            # point.time_from_start.secs = int(t//1)
            # point.time_from_start.nsecs = int((t % 1)*1e9 )
            # plan.joint_trajectory.points.append(point)
        



        # print(4)
        # kuka.display_trajectory(cartesian_plan)

        # print(5)
        kuka.execute_plan(plan)

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    

if __name__ == "__main__":
    main()