import sys, os
sys.path.append('/home/embodied_ai/lyh/act')
sys.path.append('/home/embodied_ai/catkin_ws/src/suhan_robot_model_tools')

from moveit_msgs.msg import CollisionObject, DisplayTrajectory, RobotState, RobotTrajectory
from geometry_msgs.msg import Pose, PolygonStamped, Point, Transform
from sensor_msgs.msg import JointState, Image
import std_msgs.msg as std_msg
from scipy.spatial.transform import Rotation
import numpy as np
import rospy
import os
from transform import TF_mat
import traj_gen
import message_filters
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

import argparse
import pickle
import torch
from einops import rearrange
from policy import ACTPolicy
from constants import SIM_TASK_CONFIGS

from srmt.planning_scene import PlanningScene

robot_state = RobotState()
robot_state.joint_state.header.frame_id = 'world'
robot_state.multi_dof_joint_state.header.frame_id = 'world'
robot_state.multi_dof_joint_state.joint_names.append('world_vjoint')
robot_state.multi_dof_joint_state.transforms.append(Transform())
bridge = CvBridge()

class TocabiAct:
    def __init__(self, args):
        is_eval = args['eval']
        ckpt_dir = args['ckpt_dir']
        policy_class = args['policy_class']
        onscreen_render = args['onscreen_render']
        task_name = args['task_name']
        batch_size_train = args['batch_size']
        batch_size_val = args['batch_size']
        num_epochs = args['num_epochs']
        self.temporal_agg = args['temporal_agg']
        print(self.temporal_agg)

        task_config = SIM_TASK_CONFIGS[task_name]
        dataset_dir = task_config['dataset_dir']
        num_episodes = task_config['num_episodes']
        episode_len = task_config['episode_len']
        camera_names = task_config['camera_names']

        ckpt_name = 'policy_best.ckpt'
        self.state_dim = 8
        self.max_timesteps = int(episode_len * 1.5)

        policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': 1e-5,
                        'backbone': 'resnet18',
                        'enc_layers': 4,
                        'dec_layers': 7,
                        'nheads': 8,
                        'camera_names': camera_names,
                        }

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = policy_config['num_queries']
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()
        else:
            self.query_frequency = policy_config['num_queries']

        self.t = 0
        self.terminate = True
        self.all_actions = None

        # self.planning_scene = PlanningScene(['pelvis_pose', 'whole_body', 'right_hand'], [7, 33, 20], topic_name='mujoco_tocabi', base_link='/world')
        self.joint_target_pub = rospy.Publisher("tocabi/act/joint_target", JointState, queue_size=1)
        self.traj_vis_pub = rospy.Publisher("/tocabi/act/traj_vis", DisplayTrajectory, queue_size=1)


    def sync_callback(self, point_msg, joint_msg, hand_msg, img_msg):
        pelvis_pos = point_msg.polygon.points[3]
        pelvis_rpy = point_msg.polygon.points[4]
        pelvis_quat = Rotation.from_euler('xyz', [pelvis_rpy.x, pelvis_rpy.y, pelvis_rpy.z]).as_quat()
        q_body = joint_msg.position
        q_hand = hand_msg.position

        pelvis_current= np.array([pelvis_pos.x, pelvis_pos.y, pelvis_pos.z, *pelvis_quat])
        q_current = np.array([*q_body])
        hand_current = np.array([q_hand[5], q_hand[9], *q_hand[6:9],
                                q_hand[0], q_hand[4], *q_hand[1:4],
                                q_hand[10], q_hand[14], *q_hand[11:14],
                                q_hand[15], q_hand[19], *q_hand[16:19]])

        q = np.concatenate([pelvis_current, q_current, hand_current])

        robot_state.joint_state.name = [*joint_msg.name, *hand_msg.name]
        robot_state.joint_state.position = [*q_body, *q_hand]
        robot_state.multi_dof_joint_state.transforms[0].translation.x = pelvis_current[0]
        robot_state.multi_dof_joint_state.transforms[0].translation.y = pelvis_current[1]
        robot_state.multi_dof_joint_state.transforms[0].translation.z = pelvis_current[2]
        robot_state.multi_dof_joint_state.transforms[0].rotation.x = pelvis_current[3]
        robot_state.multi_dof_joint_state.transforms[0].rotation.y = pelvis_current[4]
        robot_state.multi_dof_joint_state.transforms[0].rotation.z = pelvis_current[5]
        robot_state.multi_dof_joint_state.transforms[0].rotation.w = pelvis_current[6]
        # save_joints(f1, joint_msg)

        top_image = bridge.imgmsg_to_cv2(img_msg)
        top_image = rearrange(top_image, 'h w c -> c h w')
        curr_image = np.stack([top_image], axis=0)

        if not self.terminate:
            with torch.inference_mode():
                qpos_numpy = q_current[25:33]
                qpos = self.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

                ### query policy
                if self.t % self.query_frequency == 0:
                    self.all_actions = self.policy(qpos, curr_image)
                if self.temporal_agg:
                    self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                    actions_for_curr_step = self.all_time_actions[:, self.t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = self.all_actions[:, self.t % self.query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                target_qpos = action

                ### step the environment
                print(self.t, ': ', target_qpos)
                joint_target = JointState()
                joint_target.header = img_msg.header
                joint_target.position = target_qpos
                self.joint_target_pub.publish(joint_target)
                self.t += 1

                # q[32:40] = target_qpos
                # self.planning_scene.display(q)

    def terminate_callback(self, msg):
        self.terminate = msg.data
        self.t = 0

'''
run with following args
python tocabi_act.py \
--task_name sim_tocabi_approach_cup --ckpt_dir /home/embodied_ai/lyh/act/ckpt \
--policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 \
--dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --temporal_agg
'''
if __name__ == '__main__':
    rospy.init_node('tocabi_act', anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    tocabi_act = TocabiAct(vars(parser.parse_args()))

    point_sub = message_filters.Subscriber("/tocabi/point", PolygonStamped)
    joint_sub = message_filters.Subscriber("/tocabi/jointstates", JointState)
    hand_sub = message_filters.Subscriber("/tocabi/handstates", JointState)
    img_sub = message_filters.Subscriber("/mujoco_ros_interface/camera/image", Image)
    ts = message_filters.ApproximateTimeSynchronizer([point_sub, joint_sub, hand_sub, img_sub], 10, 0.001)
    ts.registerCallback(tocabi_act.sync_callback)

    terminate_sub = rospy.Subscriber("/tocabi/act/terminate", std_msg.Bool, tocabi_act.terminate_callback)

    rospy.spin()

