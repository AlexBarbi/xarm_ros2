#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Isaac Sim variant of _robot_moveit_gazebo.launch.py.
# ros2_control runs topic_based_ros2_control/TopicBasedSystem, which talks to Isaac Sim through
# JointState topics (/isaac_joint_commands -> Isaac, Isaac -> /isaac_joint_states).
# The Isaac Sim side is isaac/xarm_isaac_sim.py, copied with URDF and meshes into a bundle folder
# (default /tmp/xarm_isaac) and started here from a host install (isaac_sim_path) or a container
# (isaac_docker_image).

import os
import re
import shlex
import shutil
import tempfile
import xml.etree.ElementTree as ET
import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, ExecuteProcess, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file

TOPIC_BASED_PLUGIN = 'topic_based_ros2_control/TopicBasedSystem'
ISAAC_SCRIPT = 'xarm_isaac_sim.py'
ISAAC_DOCKER_DIR = '/xarm_isaac'


def add_topic_params(urdf_xml, joint_commands_topic, joint_states_topic):
    # point every TopicBasedSystem block (arm and gripper) at the Isaac Sim topics
    root = ET.fromstring(urdf_xml)
    for ros2_control in root.iter('ros2_control'):
        hardware = ros2_control.find('hardware')
        plugin = hardware.find('plugin') if hardware is not None else None
        if plugin is None or plugin.text.strip() != TOPIC_BASED_PLUGIN:
            continue
        # TopicBasedSystem names its node after the block, xacro uses the plugin name ('/' is invalid there)
        ros2_control.set('name', re.sub(r'\W', '_', ros2_control.get('name')))
        for name, value in (('joint_commands_topic', joint_commands_topic), ('joint_states_topic', joint_states_topic)):
            ET.SubElement(hardware, 'param', name=name).text = value
    return ET.tostring(root, encoding='unicode')


def write_isaac_bundle(urdf_xml, urdf_path):
    # Self-contained folder for Isaac Sim on the host or in a container: URDF without ros2_control/gazebo
    # tags and 'world' link (fix_base pins link_base at the origin), meshes copied next to it with
    # relative paths, the Isaac Sim script and the UDP-only Fast DDS profile
    bundle_dir = os.path.dirname(os.path.abspath(urdf_path))
    os.makedirs(bundle_dir, exist_ok=True)
    root = ET.fromstring(urdf_xml)
    for tag in ('ros2_control', 'gazebo', 'transmission'):
        for elem in root.findall(tag):
            root.remove(elem)
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        if parent is not None and parent.get('link') == 'world':
            root.remove(joint)
    for link in root.findall('link'):
        if link.get('name') == 'world':
            root.remove(link)
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith('package://'):
            pkg, _, rel_path = filename[len('package://'):].partition('/')
            src = os.path.join(get_package_share_directory(pkg), rel_path)
            dst = os.path.join('meshes', pkg, rel_path)
        else:
            src = filename[len('file://'):] if filename.startswith('file://') else filename
            dst = os.path.join('meshes', 'other', os.path.basename(src))
        os.makedirs(os.path.join(bundle_dir, os.path.dirname(dst)), exist_ok=True)
        shutil.copyfile(src, os.path.join(bundle_dir, dst))
        mesh.set('filename', dst)
    ET.ElementTree(root).write(urdf_path, encoding='utf-8', xml_declaration=True)
    isaac_share = os.path.join(get_package_share_directory('xarm_moveit_config'), 'isaac')
    for name in (ISAAC_SCRIPT, 'fastdds_udp.xml'):
        shutil.copyfile(os.path.join(isaac_share, name), os.path.join(bundle_dir, name))
    return bundle_dir


def isaac_docker_cmd(image, bundle_dir, script_args, ros_distro, show_gui, extra_args):
    cmd = [
        'docker', 'run', '--rm', '--name', 'xarm_isaac_sim', '--gpus', 'all', '--network=host',
        '-e', 'ACCEPT_EULA=Y', '-e', 'PRIVACY_CONSENT=Y',
        '-e', 'ROS_DISTRO={}'.format(ros_distro),
        '-e', 'RMW_IMPLEMENTATION=rmw_fastrtps_cpp',
        # bundled ROS 2 libraries: isaacsim.ros2.core in Isaac Sim 6.x, isaacsim.ros2.bridge in 5.x
        '-e', 'LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/{0}/lib:/isaac-sim/exts/isaacsim.ros2.bridge/{0}/lib'.format(ros_distro),
        '-e', 'ROS_DOMAIN_ID={}'.format(os.environ.get('ROS_DOMAIN_ID', '0')),
        # host networking makes Fast DDS pick shared memory, which does not reach the host from the container
        '-e', 'FASTRTPS_DEFAULT_PROFILES_FILE={}/fastdds_udp.xml'.format(ISAAC_DOCKER_DIR),
        '-v', '{}:{}'.format(bundle_dir, ISAAC_DOCKER_DIR),
    ]
    if show_gui and os.environ.get('DISPLAY'):
        cmd += ['-e', 'DISPLAY', '-v', '/tmp/.X11-unix:/tmp/.X11-unix']
    # no shell here, expand ~ and $VARS of e.g. cache volume mounts ourselves
    cmd += [os.path.expanduser(os.path.expandvars(arg)) for arg in shlex.split(extra_args)]
    cmd += ['--entrypoint', '/isaac-sim/python.sh', image, '{}/{}'.format(ISAAC_DOCKER_DIR, ISAAC_SCRIPT)] + script_args
    return cmd


def launch_setup(context, *args, **kwargs):
    dof = LaunchConfiguration('dof', default=7)
    robot_type = LaunchConfiguration('robot_type', default='xarm')
    prefix = LaunchConfiguration('prefix', default='')
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')
    limited = LaunchConfiguration('limited', default=True)
    effort_control = LaunchConfiguration('effort_control', default=False)
    velocity_control = LaunchConfiguration('velocity_control', default=False)
    model1300 = LaunchConfiguration('model1300', default=False)
    robot_sn = LaunchConfiguration('robot_sn', default='')
    attach_to = LaunchConfiguration('attach_to', default='world')
    attach_xyz = LaunchConfiguration('attach_xyz', default='"0 0 0"')
    attach_rpy = LaunchConfiguration('attach_rpy', default='"0 0 0"')
    mesh_suffix = LaunchConfiguration('mesh_suffix', default='stl')
    kinematics_suffix = LaunchConfiguration('kinematics_suffix', default='')
    gripper_version = LaunchConfiguration('gripper_version', default='G1')

    add_gripper = LaunchConfiguration('add_gripper', default=True)
    add_vacuum_gripper = LaunchConfiguration('add_vacuum_gripper', default=False)
    add_bio_gripper = LaunchConfiguration('add_bio_gripper', default=False)
    add_realsense_d435i = LaunchConfiguration('add_realsense_d435i', default=True)
    add_d435i_links = LaunchConfiguration('add_d435i_links', default=True)
    add_other_geometry = LaunchConfiguration('add_other_geometry', default=False)
    geometry_type = LaunchConfiguration('geometry_type', default='box')
    geometry_mass = LaunchConfiguration('geometry_mass', default=0.1)
    geometry_height = LaunchConfiguration('geometry_height', default=0.1)
    geometry_radius = LaunchConfiguration('geometry_radius', default=0.1)
    geometry_length = LaunchConfiguration('geometry_length', default=0.1)
    geometry_width = LaunchConfiguration('geometry_width', default=0.1)
    geometry_mesh_filename = LaunchConfiguration('geometry_mesh_filename', default='')
    geometry_mesh_origin_xyz = LaunchConfiguration('geometry_mesh_origin_xyz', default='"0 0 0"')
    geometry_mesh_origin_rpy = LaunchConfiguration('geometry_mesh_origin_rpy', default='"0 0 0"')
    geometry_mesh_tcp_xyz = LaunchConfiguration('geometry_mesh_tcp_xyz', default='"0 0 0"')
    geometry_mesh_tcp_rpy = LaunchConfiguration('geometry_mesh_tcp_rpy', default='"0 0 0"')

    no_gui_ctrl = LaunchConfiguration('no_gui_ctrl', default=False)
    show_rviz = LaunchConfiguration('show_rviz', default=True)
    ros_namespace = LaunchConfiguration('ros_namespace', default='').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True).perform(context) in ('True', 'true')
    update_rate = int(LaunchConfiguration('update_rate', default=150).perform(context))

    # Isaac Sim
    joint_commands_topic = LaunchConfiguration('isaac_joint_commands_topic', default='/isaac_joint_commands').perform(context)
    joint_states_topic = LaunchConfiguration('isaac_joint_states_topic', default='/isaac_joint_states').perform(context)
    isaac_sim_path = LaunchConfiguration('isaac_sim_path', default=os.environ.get('ISAAC_SIM_PATH', '')).perform(context)
    isaac_headless = LaunchConfiguration('isaac_headless', default=False).perform(context) in ('True', 'true')
    isaac_livestream = LaunchConfiguration('isaac_livestream', default=False).perform(context) in ('True', 'true')
    isaac_table = LaunchConfiguration('isaac_table', default=True).perform(context) in ('True', 'true')
    # Isaac Sim container, e.g. nvcr.io/nvidia/isaac-sim:5.1.0, takes precedence over isaac_sim_path
    isaac_docker_image = LaunchConfiguration('isaac_docker_image', default='').perform(context)
    isaac_docker_args = LaunchConfiguration('isaac_docker_args', default='').perform(context)
    isaac_ros_distro = LaunchConfiguration('isaac_ros_distro', default='jazzy').perform(context)

    ros2_control_plugin = TOPIC_BASED_PLUGIN
    controllers_name = 'fake_controllers'
    xarm_type = '{}{}'.format(robot_type.perform(context), dof.perform(context) if robot_type.perform(context) in ('xarm', 'lite') else '')
    isaac_urdf_path = LaunchConfiguration('isaac_urdf_path', default=os.path.join(tempfile.gettempdir(), 'xarm_isaac', '{}.urdf'.format(xarm_type))).perform(context)

    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(get_package_share_directory('xarm_controller'), 'config', '{}_controllers.yaml'.format(xarm_type)),
        prefix=prefix.perform(context),
        add_gripper=add_gripper.perform(context) in ('True', 'true'),
        add_bio_gripper=add_bio_gripper.perform(context) in ('True', 'true'),
        ros_namespace=ros_namespace,
        update_rate=update_rate,
        use_sim_time=use_sim_time,
        robot_type=robot_type.perform(context)
    )

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name=controllers_name,
        dof=dof,
        robot_type=robot_type,
        prefix=prefix,
        hw_ns=hw_ns,
        limited=limited,
        effort_control=effort_control,
        velocity_control=velocity_control,
        model1300=model1300,
        robot_sn=robot_sn,
        attach_to=attach_to,
        attach_xyz=attach_xyz,
        attach_rpy=attach_rpy,
        mesh_suffix=mesh_suffix,
        kinematics_suffix=kinematics_suffix,
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        gripper_version=gripper_version,
        add_gripper=add_gripper,
        add_vacuum_gripper=add_vacuum_gripper,
        add_bio_gripper=add_bio_gripper,
        add_realsense_d435i=add_realsense_d435i,
        add_d435i_links=add_d435i_links,
        add_other_geometry=add_other_geometry,
        geometry_type=geometry_type,
        geometry_mass=geometry_mass,
        geometry_height=geometry_height,
        geometry_radius=geometry_radius,
        geometry_length=geometry_length,
        geometry_width=geometry_width,
        geometry_mesh_filename=geometry_mesh_filename,
        geometry_mesh_origin_xyz=geometry_mesh_origin_xyz,
        geometry_mesh_origin_rpy=geometry_mesh_origin_rpy,
        geometry_mesh_tcp_xyz=geometry_mesh_tcp_xyz,
        geometry_mesh_tcp_rpy=geometry_mesh_tcp_rpy,
    ).to_moveit_configs()

    urdf_xml = add_topic_params(moveit_config.robot_description['robot_description'], joint_commands_topic, joint_states_topic)
    moveit_config.robot_description = {'robot_description': urdf_xml}
    isaac_bundle_dir = write_isaac_bundle(urdf_xml, isaac_urdf_path)

    moveit_config_dump = yaml.dump(moveit_config.to_dict())

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, moveit_config.robot_description],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ]
    )

    # robot moveit common launch
    # xarm_moveit_config/launch/_robot_moveit_common2.launch.py
    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py'])),
        launch_arguments={
            'prefix': prefix,
            'attach_to': attach_to,
            'attach_xyz': attach_xyz,
            'attach_rpy': attach_rpy,
            'no_gui_ctrl': no_gui_ctrl,
            'show_rviz': show_rviz,
            'use_sim_time': str(use_sim_time).lower(),
            'moveit_config_dump': moveit_config_dump,
        }.items(),
    )

    # ros2 control launch
    # xarm_controller/launch/_ros2_control.launch.py
    ros2_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_controller'), 'launch', '_ros2_control.launch.py'])),
        launch_arguments={
            'robot_description': yaml.dump(moveit_config.robot_description),
            'ros2_control_params': ros2_control_params,
        }.items(),
    )

    controllers = ['joint_state_broadcaster', '{}{}_traj_controller'.format(prefix.perform(context), xarm_type)]
    if add_gripper.perform(context) in ('True', 'true'):
        controllers.append('{}{}_gripper_traj_controller'.format(prefix.perform(context), robot_type.perform(context)))
    elif add_bio_gripper.perform(context) in ('True', 'true') and robot_type.perform(context) != 'lite':
        controllers.append('{}bio_gripper_traj_controller'.format(prefix.perform(context)))

    # one spawner for all controllers, separate spawners queue on a lock that gives up before a slow Isaac Sim start
    controller_nodes = [Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=controllers + [
            '--controller-manager', '{}/controller_manager'.format(ros_namespace),
            # with sim time the controller manager only cycles once Isaac Sim publishes /clock,
            # which takes a while on startup (default switch timeout is 5 s)
            '--switch-timeout', '300',
            '--service-call-timeout', '310',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )]

    # Isaac Sim scene: container (isaac_docker_image), host install (isaac_sim_path / $ISAAC_SIM_PATH),
    # or started by hand from the bundle folder
    isaac_flags = [
        '--joint-commands-topic', joint_commands_topic.lstrip('/'),
        '--joint-states-topic', joint_states_topic.lstrip('/'),
    ]
    if isaac_headless:
        isaac_flags.append('--headless')
    if isaac_livestream:
        isaac_flags.append('--livestream')
    if not isaac_table:
        isaac_flags.append('--no-table')
    host_cmd = [
        os.path.join(isaac_bundle_dir, ISAAC_SCRIPT), '--urdf', os.path.abspath(isaac_urdf_path)
    ] + isaac_flags
    if isaac_docker_image:
        docker_flags = ['--urdf', '{}/{}'.format(ISAAC_DOCKER_DIR, os.path.basename(isaac_urdf_path))] + isaac_flags
        isaac_node = ExecuteProcess(
            cmd=isaac_docker_cmd(isaac_docker_image, isaac_bundle_dir, docker_flags, isaac_ros_distro,
                                 not (isaac_headless or isaac_livestream), isaac_docker_args),
            output='screen',
        )
    elif isaac_sim_path:
        # Isaac Sim runs its own python; system ROS python packages on PYTHONPATH break it
        isaac_env = dict(os.environ)
        isaac_env.pop('PYTHONPATH', None)
        isaac_node = ExecuteProcess(
            cmd=[os.path.join(os.path.expanduser(isaac_sim_path), 'python.sh')] + host_cmd,
            env=isaac_env,
            output='screen',
        )
    else:
        isaac_node = LogInfo(msg='Isaac Sim not started (set isaac_sim_path or isaac_docker_image), run it yourself:\n'
                                 '  <isaac_sim>/python.sh {}'.format(' '.join(host_cmd)))

    return [
        isaac_node,
        robot_state_publisher_node,
        robot_moveit_common_launch,
        ros2_control_launch,
    ] + controller_nodes


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
