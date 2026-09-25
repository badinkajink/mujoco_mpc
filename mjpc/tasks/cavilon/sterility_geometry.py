"""Shared task geometry and whole-arm collision contract for the Cavilon pilot.

Distances are signed MuJoCo geom distances, including visual convex meshes for
the site cylinder. Nothing is exempt from that cylinder except its own marker.
The cylinder is along task +Z (the reconstructed patch's mean normal), from
20 mm below the site to 2 m above it. Its top is above the arm's full reach.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from build_contact_replay import load_geometry, split_segments, write_hfield_png
from render_flat_pad_augmented import insertion_site_local
from render_ur5_pad_feasibility import build_scene

ORIGIN = np.array([.55, 0., .25])
EXCLUSION_RADIUS = .0254
MARGIN = .004
PAD_HALF = np.array([.010, .0045, .002])


@dataclass
class Target:
    position: np.ndarray
    rotation: np.ndarray
    phase: str
    segment: int
    original: np.ndarray


def scene(root: Path, ur5: Path, output: Path, offset: float = .100):
    output.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(root, 128)
    site, provenance = insertion_site_local(root, geometry)
    hfield = output / 'mannequin.png'
    write_hfield_png(geometry['height'], hfield)
    destination = output / 'scene.xml'
    build_scene(ur5, hfield, geometry, destination)
    tree = ET.parse(destination); xml = tree.getroot()
    xml.set('model', 'UR5e whole-arm insertion-site exclusion')
    # Explicitly name every geometry so failures identify a physical part.
    for body in xml.iter('body'):
        for i, geom in enumerate(body.findall('geom')):
            if not geom.get('name'):
                geom.set('name', f"{body.get('name')}_{geom.get('class', 'geom')}_{i}")
    world = xml.find('worldbody')
    patient = world.find("geom[@name='pid117_mannequin']")
    patient.set('contype', '1'); patient.set('conaffinity', '1')
    tool = next(b for b in xml.iter('body') if b.get('name') == 'cavilon_tool')
    for child in list(tool): tool.remove(child)
    # A narrow dogleg places the wrist radially outward while the pad is flat.
    # Dimensions are design variables, not measured applicator geometry.
    reach = .160 if offset else .110
    z_bend = reach - .075
    common = {'contype': '1', 'conaffinity': '1'}
    for name, ends in [('tool_stem', f'0 0 .012 0 0 {z_bend}'),
                       ('tool_offset', f'0 0 {z_bend} 0 {offset} {z_bend}'),
                       ('tool_neck', f'0 {offset} {z_bend} 0 {offset} {reach-.005}')]:
        if name == 'tool_offset' and not offset: continue
        ET.SubElement(tool, 'geom', dict(name=name, type='capsule', fromto=ends,
                      size='.0025', rgba='.82 .88 .96 1', **common))
    ET.SubElement(tool, 'geom', dict(name='tool_backing', type='box',
                  pos=f'0 {offset} {reach-.005}', size='.0105 .005 .001',
                  rgba='.9 .9 .9 1', **common))
    ET.SubElement(tool, 'geom', dict(name='tool_pad', type='box',
                  pos=f'0 {offset} {reach-.002}', size='.010 .0045 .002',
                  rgba='.95 .76 .25 1', **common))
    ET.SubElement(tool, 'site', dict(name='cavilon_contact', pos=f'0 {offset} {reach}',
                                   size='.001', rgba='.1 .9 .8 1'))
    c = site + ORIGIN
    ET.SubElement(world, 'geom', dict(name='site_exclusion', type='cylinder',
                  pos=f'{c[0]} {c[1]} {c[2]+.99}', size=f'{EXCLUSION_RADIUS} 1.01',
                  rgba='1 .12 .06 .17', contype='0', conaffinity='0', group='1'))
    ET.SubElement(world, 'geom', dict(name='site_marker', type='cylinder',
                  pos=f'{c[0]} {c[1]} {c[2]+.0005}', size='.004 .0005',
                  rgba='1 .1 .05 1', contype='0', conaffinity='0'))
    tree.write(destination, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(destination))
    return model, geometry, site, provenance, destination


class CollisionContract:
    """Shared by IK, OMPL, and independent dense final validation."""
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.geom('site_exclusion').id
        self.floor_id = model.geom('floor').id
        self.patient_id = model.geom('pid117_mannequin').id
        self.tool_body = model.body('cavilon_tool').id
        self.base_body = model.body('base').id
        def belongs_to_robot(body):
            while body:
                if body == self.base_body:return True
                body = model.body_parentid[body]
            return False
        self.robot = [i for i in range(model.ngeom) if belongs_to_robot(model.geom_bodyid[i])]
        self.primitive = [i for i in self.robot if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH]
        self.cylinder_pairs = [(i, self.site_id) for i in self.robot]
        self.floor_pairs = [(i, self.floor_id) for i in self.robot
                            if model.geom_bodyid[i] != self.base_body]
        self.self_pairs = []
        for n, a in enumerate(self.primitive):
            for b in self.primitive[n+1:]:
                ba, bb = model.geom_bodyid[[a, b]]
                if ba == bb or model.body_parentid[ba] == bb or model.body_parentid[bb] == ba:
                    continue  # mechanically attached/adjacent link envelopes
                self.self_pairs.append((a, b))
        self.pairs = self.cylinder_pairs + self.floor_pairs + self.self_pairs
        self.groups = [([a], [b]) for a, b in self.pairs]
        self.checks = 0
        self.projected_vertices = {}
        for i in self.robot:
            kind=model.geom_type[i];size=model.geom_size[i]
            if kind==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=model.geom_dataid[i];a=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
                vertices=model.mesh_vert[a:a+n].copy()
                vertices=vertices[ConvexHull(vertices).vertices]
            elif kind==mujoco.mjtGeom.mjGEOM_BOX:
                vertices=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*size
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
                angles=np.linspace(0,2*np.pi,64,endpoint=False)
                radius=size[0]/np.cos(np.pi/64)  # circumscribed, conservative
                vertices=np.array([[radius*np.cos(a),radius*np.sin(a),z]
                                   for z in [-size[1],size[1]] for a in angles])
            else:continue
            self.projected_vertices[i]=vertices
        # Conservative bound on distance of any robot point from any ancestor
        # revolute joint. Triangle inequality over fixed body translations is
        # valid for all configurations, not only the current pose.
        self.point_motion_radius=0.
        for i in self.robot:
            body=int(model.geom_bodyid[i]);reach=float(np.linalg.norm(model.geom_pos[i])+model.geom_rbound[i])
            while body:
                for joint in range(model.body_jntadr[body],model.body_jntadr[body]+model.body_jntnum[body]):
                    if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError('Point-motion certificate currently assumes revolute robot joints')
                    self.point_motion_radius=max(self.point_motion_radius,reach+float(np.linalg.norm(model.jnt_pos[joint])))
                reach+=float(np.linalg.norm(model.body_pos[body]));body=int(model.body_parentid[body])

    def projected_cylinder_clearance(self, q):
        """Independent conservative 2D convex projection vs infinite cylinder.

        Stronger than the finite 2 m volume, and independent of MuJoCo GJK.
        Cylinders use a circumscribed 64-gon; boxes/meshes and capsules use
        exact projected convex hulls / capsule disks. Returns a lower bound.
        """
        self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data)
        center=self.data.geom_xpos[self.site_id,:2];radius=self.model.geom_size[self.site_id,0]
        distances=[]
        for i in self.robot:
            kind=self.model.geom_type[i];size=self.model.geom_size[i]
            position=self.data.geom_xpos[i];rotation=self.data.geom_xmat[i].reshape(3,3)
            if kind in (mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE):
                dz=rotation[:2,2]*size[1] if kind==mujoco.mjtGeom.mjGEOM_CAPSULE else np.zeros(2)
                a=position[:2]-dz;b=position[:2]+dz;delta=b-a
                alpha=np.clip((center-a)@delta/max(delta@delta,1e-30),0,1)
                distance=np.linalg.norm(center-a-alpha*delta)-size[0]-radius
            else:
                points=(self.projected_vertices[i]@rotation.T+position)[:,:2]
                polygon=points[ConvexHull(points).vertices];edges=np.roll(polygon,-1,axis=0)-polygon
                offset=center-polygon
                alpha=np.clip(np.sum(offset*edges,axis=1)/np.sum(edges*edges,axis=1),0,1)
                edge_distance=np.linalg.norm(offset-alpha[:,None]*edges,axis=1).min()
                inside=np.all(edges[:,0]*offset[:,1]-edges[:,1]*offset[:,0]>=-1e-12)
                distance=(-edge_distance if inside else edge_distance)-radius
            distances.append(distance)
        index=int(np.argmin(distances))
        return float(distances[index]),self.model.geom(self.robot[index]).name

    def distances(self, q, max_distance=.10):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.checks += 1
        result=np.array([mujoco.mj_geomDistance(self.model, self.data, a, b, max_distance, None)
                         for a, b in self.pairs])
        for j,(a,b) in enumerate(self.cylinder_pairs):
            if result[j]==0 and self.model.geom_type[a] in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_SPHERE):
                size=self.model.geom_size[a];rotation=self.data.geom_xmat[a].reshape(3,3)
                dz=rotation[:2,2]*size[1] if self.model.geom_type[a]==mujoco.mjtGeom.mjGEOM_CAPSULE else np.zeros(2)
                p=self.data.geom_xpos[a,:2]-dz;delta=2*dz;center=self.data.geom_xpos[b,:2]
                alpha=np.clip((center-p)@delta/max(delta@delta,1e-30),0,1)
                result[j]=min(max_distance,np.linalg.norm(center-p-alpha*delta)-size[0]-self.model.geom_size[b,0])
            if (self.model.geom_type[a]==mujoco.mjtGeom.mjGEOM_CYLINDER
                    or (result[j]==0 and a in self.projected_vertices)):
                # MuJoCo 3.10 can return spurious zero for separated long
                # cylinder/convex pairs. A conservative planar projection
                # provides a fail-closed distance for the infinite cylinder.
                vertices=self.projected_vertices[a]
                rotation=self.data.geom_xmat[a].reshape(3,3)
                points=(vertices@rotation.T+self.data.geom_xpos[a])[:,:2]
                polygon=points[ConvexHull(points).vertices]
                edges=np.roll(polygon,-1,axis=0)-polygon
                offset=self.data.geom_xpos[b,:2]-polygon
                alpha=np.clip(np.sum(offset*edges,axis=1)/np.sum(edges*edges,axis=1),0,1)
                distance=np.linalg.norm(offset-alpha[:,None]*edges,axis=1).min()
                inside=np.all(edges[:,0]*offset[:,1]-edges[:,1]*offset[:,0]>=-1e-12)
                result[j]=min(max_distance,(-distance if inside else distance)-self.model.geom_size[b,0])
        return result

    def patient_penetration(self):
        # mj_geomDistance is for convex pairs. Hfield contact must be read from
        # the collision manifold, not inferred from a distmax sentinel.
        depth = 0.
        for contact in self.data.contact:
            if self.patient_id in (contact.geom1, contact.geom2):
                depth = max(depth, -float(contact.dist))
        return depth

    def valid(self, q, clearance=.001):
        if np.any(q < self.model.jnt_range[:, 0]) or np.any(q > self.model.jnt_range[:, 1]):
            return False
        distances = self.distances(q)
        n = len(self.cylinder_pairs)
        return bool(distances[:n].min() >= clearance and distances[n:].min() >= .001
                    and self.patient_penetration() <= .00005)

    def summary(self, q):
        # Large GJK cutoffs can return a spurious zero for separated long
        # cylinder pairs in MuJoCo 3.10. Match the validity check's 0.1 m
        # horizon; larger clearances are right-censored, not measured exactly.
        distances = self.distances(q, .10)
        n, f = len(self.cylinder_pairs), len(self.floor_pairs)
        i = int(np.argmin(distances[:n]))
        patient_contacts = []
        for contact in self.data.contact:
            if self.patient_id in (contact.geom1, contact.geom2) and contact.dist < 0:
                other = contact.geom2 if contact.geom1 == self.patient_id else contact.geom1
                patient_contacts.append(self.model.geom(other).name)
        return dict(cylinder_mm=float(distances[:n].min()*1000),
                    floor_mm=float(distances[n:n+f].min()*1000),
                    self_mm=float(distances[n+f:].min()*1000),
                    patient_penetration_mm=float(self.patient_penetration()*1000),
                    closest_geom=self.model.geom(self.cylinder_pairs[i][0]).name,
                    patient_contacts=';'.join(sorted(set(patient_contacts))))


def make_targets(geometry, site, spacing=.002):
    gx, gy, heights = geometry['gx'], geometry['gy'], geometry['height']
    surface = RegularGridInterpolator((gy, gx), heights, bounds_error=True)
    # The finite pad follows a pad-scale surface normal, not pixel-scale depth
    # corrugations. Terrain collision still uses the unmodified heightfield.
    dhdy, dhdx = np.gradient(gaussian_filter(heights, sigma=3.), gy, gx)
    nx = RegularGridInterpolator((gy, gx), dhdx, bounds_error=True)
    ny = RegularGridInterpolator((gy, gx), dhdy, bounds_error=True)
    # Circumscribed pad sphere supplies a conservative XY inflation even on
    # slopes. Exact robot/cylinder distances remain the deciding constraint.
    radius = EXCLUSION_RADIUS + np.linalg.norm(PAD_HALF) + MARGIN + .002

    def pose(xy, lift, phase, segment, original):
        z = float(surface([[xy[1], xy[0]]])[0])
        normal = np.array([-float(nx([[xy[1], xy[0]]])[0]),
                           -float(ny([[xy[1], xy[0]]])[0]), 1.])
        normal /= np.linalg.norm(normal)
        radial = np.r_[xy-site[:2], 0.]
        radial -= normal*np.dot(radial, normal); radial /= np.linalg.norm(radial)
        y_axis = -radial; z_axis = -normal
        x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        # Finite planar pad on a curved surface: lift its face until all four
        # corners are on/above terrain. Contact mechanics is a subsequent task.
        # In-plane yaw is free in IK. Bound terrain support for every yaw using
        # the pad's circumscribed disk instead of only one rectangle orientation.
        disk_radius=np.hypot(.010,.0045)
        offsets=np.array([[r*np.cos(a),r*np.sin(a),0.]
                          for r in np.linspace(0,disk_radius,8)
                          for a in np.linspace(0,2*np.pi,48,endpoint=False)]) @ rotation.T
        corners = np.r_[xy, z] + offsets
        clearance = surface(corners[:, [1, 0]]) - corners[:, 2]
        face_lift = max(0., float(clearance.max())) + .0008
        position = ORIGIN + np.r_[xy, z + face_lift + lift]
        return Target(position, rotation, phase, segment, original)

    segments, adjustments = [], []
    for number, (a, b) in enumerate(split_segments(geometry['path']['frame']), 1):
        original = np.asarray(geometry['path_xy'][a:b])
        smooth = gaussian_filter1d(original, 2.0, axis=0, mode='nearest')
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
        t = np.linspace(0, distance[-1], max(3, int(np.ceil(distance[-1]/spacing))+1))
        xy = np.column_stack([np.interp(t, distance, smooth[:, d]) for d in (0, 1)])
        delta = xy-site[:2]; r = np.linalg.norm(delta, axis=1)
        safe = site[:2] + delta*np.maximum(1., radius/r)[:, None]
        adjustments.extend(np.linalg.norm(safe-xy, axis=1).tolist())
        segments.append([pose(p, 0., 'surface_follow', number, raw) for p, raw in zip(safe, xy)])

    targets = []
    for number, segment in enumerate(segments):
        first_xy = segment[0].position[:2]-ORIGIN[:2]
        if not targets:
            targets.extend(pose(first_xy, lift, 'approach', 0, first_xy)
                           for lift in np.linspace(.065, 0, 18, endpoint=False))
        else:
            last_xy = targets[-1].position[:2]-ORIGIN[:2]
            targets.extend(pose(last_xy, lift, 'lift', 0, last_xy)
                           for lift in np.linspace(0, .065, 18)[1:])
            d0, d1 = last_xy-site[:2], first_xy-site[:2]
            theta0, theta1 = np.arctan2(d0[1], d0[0]), np.arctan2(d1[1], d1[0])
            delta_angle = np.arctan2(np.sin(theta1-theta0), np.cos(theta1-theta0))
            count = max(10, int(abs(delta_angle)*max(np.linalg.norm(d0), np.linalg.norm(d1))/spacing)+1)
            for alpha in np.linspace(0, 1, count)[1:]:
                angle = theta0+alpha*delta_angle
                r = (1-alpha)*np.linalg.norm(d0)+alpha*np.linalg.norm(d1)
                xy = site[:2]+r*np.array([np.cos(angle), np.sin(angle)])
                targets.append(pose(xy, .065, 'transfer_arc', 0, xy))
            targets.extend(pose(first_xy, lift, 'descend', 0, first_xy)
                           for lift in np.linspace(.065, 0, 18)[1:-1])
        targets.extend(segment)
    last_xy = targets[-1].position[:2]-ORIGIN[:2]
    targets.extend(pose(last_xy, lift, 'retract', 0, last_xy)
                   for lift in np.linspace(0, .065, 18)[1:])
    return targets, dict(projected_center_radius_mm=radius*1000,
                         adjusted_samples=int(np.sum(np.array(adjustments)>1e-9)),
                         surface_samples=len(adjustments),
                         mean_adjustment_mm=float(np.mean(adjustments)*1000),
                         max_adjustment_mm=float(np.max(adjustments)*1000))
