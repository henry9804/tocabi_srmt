import numpy as np
import xml.etree.ElementTree as ET
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh
from geometry_msgs.msg import Pose
from utils import base_converter, vectors_to_pose

import os, sys
USERNAME = os.getlogin()
sys.path.append('/home/{}/catkin_ws/src/suhan_robot_model_tools'.format(USERNAME))
from suhan_robot_model_tools.suhan_robot_model_tools_wrapper_cpp import get_mesh_from_file

class xmlParser:
    def __init__(self, xml_name):
        tree = ET.parse(xml_name)
        root = tree.getroot()
        self.worldbody = None
        self.asset = None
        self.meshdir = None
        for child in root:
            if child.tag == 'worldbody':
                self.worldbody = child
            if child.tag == 'compiler':
                self.meshdir = os.path.join(os.path.dirname(xml_name), child.attrib['meshdir'])
            if child.tag == 'asset':
                self.asset = child
        self.meshdir = os.path.join(os.path.dirname(xml_name), '../meshes')

    def get_object_msgs(self):
        msgs = []
        for body in self.worldbody:
            if body.tag == 'body' and not body.attrib['name'].endswith('link'):
                msg = self.make_object_msg(body)
                msgs.append(msg)
        return msgs
    
    def make_object_msg(self, body, msg=None):
        for child in body:
            if child.tag == 'geom':
                if child.attrib['type'] == 'mesh':
                    mesh, pose = self.get_mesh(child.attrib)
                    if msg is None:
                        msg = CollisionObject()
                        msg.id = body.attrib['name']
                        msg.header.frame_id = 'world'
                        msg.pose = pose
                        msg.operation = msg.ADD
                        p = Pose()
                        p.orientation.w = 1
                        msg.mesh_poses.append(p)
                    else:
                        msg.mesh_poses.append(base_converter(pose, msg.pose))
                    msg.meshes.append(mesh)
                else:
                    primitive, pose = self.get_primitive(child.attrib)
                    if msg is None:
                        msg = CollisionObject()
                        msg.id = body.attrib['name']
                        msg.header.frame_id = 'world'
                        msg.pose = pose
                        msg.operation = msg.ADD
                        p = Pose()
                        p.orientation.w = 1
                        msg.primitive_poses.append(p)
                    else:
                        msg.primitive_poses.append(base_converter(pose, msg.pose))
                    msg.primitives.append(primitive)

            elif child.tag == 'body':
                self.make_object_msg(child, msg)

        return msg
    
    def get_primitive(self, attribute):
        primitive = SolidPrimitive()
        if attribute['type'] == 'box':
            primitive.type = primitive.BOX
            for s in attribute['size'].split(' '):
                primitive.dimensions.append(float(s)*2)

        elif attribute['type'] == 'sphere':
            primitive.type = primitive.SPHERE
            for s in attribute['size'].split(' '):
                primitive.dimensions.append(float(s))

        elif attribute['type'] == 'cylinder':
            primitive.type = primitive.CYLINDER
            size = attribute['size'].split(' ')
            if len(size) == 1:
                primitive.dimensions.append(float(size[0]))
            else:
                primitive.dimensions.append(float(size[1])*2)   # height
                primitive.dimensions.append(float(size[0]))     # radius

        pos = None
        ori = None
        if 'pos' in attribute:
            pos = [float(i) for i in attribute['pos'].split(' ')]
        if 'quat' in attribute:
            ori = [float(i) for i in attribute['quat'].split(' ')]
        if 'euler' in attribute:
            ori = [float(i) for i in attribute['euler'].split(' ')]
        pose = vectors_to_pose(pos, ori)

        return primitive, pose

    def get_mesh(self, attribute):
        asset_name = attribute['mesh']
        for child in self.asset:
            if child.tag == 'mesh':
                if child.attrib['name'] == asset_name:
                    meshfile = child.attrib['file']
        mesh = Mesh()
        mesh.deserialize(get_mesh_from_file(f'file://{os.path.join(self.meshdir, meshfile)}'))

        pos = None
        ori = None
        if 'pos' in attribute:
            pos = [float(i) for i in attribute['pos'].split(' ')]
        if 'quat' in attribute:
            ori = [float(i) for i in attribute['quat'].split(' ')]
        if 'euler' in attribute:
            ori = [float(i) for i in attribute['euler'].split(' ')]
        pose = vectors_to_pose(pos, ori)

        return mesh, pose