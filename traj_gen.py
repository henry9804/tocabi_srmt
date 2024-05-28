from casadi import *
import rospy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from dyros_math_python import *

def ocp_trajectory(path, joint_upper_limit, joint_lower_limit, v_max):
    num_waypoints, dof = path.shape
    # forward kinematics
    dt = SX.sym('dt')
    q = SX.sym('q', dof)
    q_dot = SX.sym('q_dot', dof)
    q_2dot = SX.sym('q_2dot', dof)
    f = Function('f', [q, q_dot, q_2dot, dt], [q + q_dot * dt + 0.5 * q_2dot * (dt ** 2), q_dot + q_2dot * dt])

    # ocp variables
    t = SX.sym('t', num_waypoints - 1)
    position = SX.sym('position', dof, num_waypoints)
    velocity = SX.sym('velocity', dof, num_waypoints)
    acceleration = SX.sym('acceleration', dof, num_waypoints)
    init_state = SX.sym('init_state', 2 * dof)
    final_state = SX.sym('final_state', 2 * dof)

    # cost & constraints
    cost = 0
    g = vertcat(position[:,0], velocity[:,0]) - init_state
    for i in range(num_waypoints-1):
        g = vertcat(g, vertcat(position[:,i+1], velocity[:,i+1]) - vertcat(*f(position[:,i], velocity[:,i], acceleration[:,i], t[i])))
        cost += (transpose(position[:,i] - path[i]) @ (position[:,i] - path[i]) + t[i] ** 2)
                #  + transpose(acceleration[:,i]) @ acceleration[:,i])
    g = vertcat(g, vertcat(position[:,-1], velocity[:,-1]) - final_state)
    
    # casadi formulation
    nlp = {}
    nlp['x'] = vertcat(reshape(position, dof * num_waypoints, 1), reshape(velocity, dof * num_waypoints, 1), reshape(acceleration, dof * num_waypoints, 1), t)
    nlp['p'] = vertcat(init_state, final_state)
    nlp['f'] = cost
    nlp['g'] = g

    opts = {}
    opts['ipopt'] = {}
    opts['ipopt']['max_iter'] = 100
    opts['ipopt']['acceptable_tol'] = 1e-8
    opts['ipopt']['acceptable_obj_change_tol'] = 1e-6
    opts['ipopt']['print_level'] = 0
    opts['print_time'] = 0

    F = nlpsol('F', 'ipopt', nlp, opts)

    args = {}
    args['x0']  = vertcat(reshape(transpose(path), dof*num_waypoints, 1), np.zeros((dof*num_waypoints, 1)), np.zeros((dof*num_waypoints, 1)), np.ones((num_waypoints - 1, 1)))
    args['p']   = vertcat(path[0], np.zeros(dof), path[-1], np.zeros(dof))
    args['ubx'] = vertcat(repmat(joint_upper_limit, num_waypoints, 1), repmat(v_max, num_waypoints, 1), np.ones(dof*num_waypoints), np.inf * np.ones(num_waypoints - 1))
    args['lbx'] = vertcat(repmat(joint_lower_limit, num_waypoints, 1), -repmat(v_max, num_waypoints, 1), -np.ones(dof*num_waypoints), np.zeros(num_waypoints - 1))
    args['ubg'] = np.zeros(2 * dof * (num_waypoints + 1))
    args['lbg'] = np.zeros(2 * dof * (num_waypoints + 1))

    # solve ocp
    solution = F(x0 = args['x0'], p = args['p'], ubx = args['ubx'], lbx = args['lbx'], ubg = args['ubg'], lbg = args['lbg'])
    opt_position = np.reshape(solution['x'][:(dof*num_waypoints)], (num_waypoints, dof))
    opt_velocity = np.reshape(solution['x'][(dof*num_waypoints):(2*dof*num_waypoints)], (num_waypoints, dof))
    opt_acceleration = np.reshape(solution['x'][(2*dof * num_waypoints):(3*dof*num_waypoints)], (num_waypoints, dof))
    opt_dt = solution['x'][-(num_waypoints-1):]
    opt_time = np.zeros(num_waypoints)
    for i in range(num_waypoints - 1):
        opt_time[i+1] = opt_time[i] + opt_dt[i]

    # make trajectory msg
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.header.seq = 0
    trajectory.joint_trajectory.header.stamp = rospy.Time(0)
    trajectory.joint_trajectory.header.frame_id = 'base'
    # trajectory.joint_trajectory.joint_names = joint_names
    t = 0
    hz = 2000
    dt = 1/hz
    for i in range(num_waypoints - 1):
        while t < opt_time[i+1]:
            point = JointTrajectoryPoint()
            point.positions = cubicVector(t, opt_time[i], opt_time[i+1], 
                                          opt_position[i], opt_position[i+1], 
                                          opt_velocity[i], opt_velocity[i+1])
            point.velocities = cubicDotVector(t, opt_time[i], opt_time[i+1], 
                                              opt_position[i], opt_position[i+1], 
                                              opt_velocity[i], opt_velocity[i+1])
            point.time_from_start = rospy.Duration(nsecs=t*1e9)
            trajectory.joint_trajectory.points.append(point)
            t += dt
    point = JointTrajectoryPoint()
    point.positions = opt_position[-1]
    point.velocities = opt_velocity[-1]
    point.time_from_start = rospy.Duration(nsecs=opt_time[-1]*1e9)
    trajectory.joint_trajectory.points.append(point)

    '''
    for pos, vel, acc, t in zip(opt_position, opt_velocity, opt_acceleration, opt_time):
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = vel
        point.accelerations = acc
        point.time_from_start = rospy.Duration(nsecs=t*1e9)
        trajectory.joint_trajectory.points.append(point)
    '''

    return trajectory