"""Regression checks for collision-pair filtering, geometry and display math."""
import unittest
import tempfile
from pathlib import Path
import mujoco
import mink
import numpy as np

from plan_sterile_ur5 import ExplicitCollisionLimit, pose_error, time_parameterize
from render_flat_pad_augmented import paint_projected_face, map_point
from sterility_geometry import scene, CollisionContract, make_targets, ORIGIN
from finalize_sterile_ur5 import certify_exclusion_edges

ROOT=Path('/home/humanoid/Programs/cldc')
UR5=Path('/home/humanoid/Programs/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/universal_robots_ur5e/ur5e.xml')


class CollisionLimitTests(unittest.TestCase):
    def test_planning_only_geometries_are_kept_and_margin_has_displacement_units(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="obstacle" type="sphere" size=".01" contype="0" conaffinity="0"/>
          <body pos=".03 0 0"><joint type="slide" axis="1 0 0"/>
            <geom name="moving" type="sphere" size=".01" contype="0" conaffinity="0" mass="1"/>
          </body></worldbody></mujoco>''')
        ids=([model.geom('obstacle').id],[model.geom('moving').id])
        config=mink.Configuration(model)
        default=mink.CollisionAvoidanceLimit(model,[ids])
        self.assertEqual(default.max_num_contacts,0)
        explicit=ExplicitCollisionLimit(model,[ids],gain=.75*.02,
                  minimum_distance_from_collisions=.004,collision_detection_distance=.04)
        self.assertEqual(explicit.max_num_contacts,1)
        constraint=explicit.compute_qp_inequalities(config,.02)
        self.assertAlmostEqual(float(constraint.h[0]),.75*.006,places=7)
        self.assertTrue(np.all(constraint.G@np.array([-.004])<=constraint.h))
        self.assertFalse(np.all(constraint.G@np.array([-.006])<=constraint.h))

    def test_fractional_simulation_clock(self):
        steps=np.array([round((i+1)/30/.001)-round(i/30/.001) for i in range(535)])
        self.assertEqual(set(steps),{33,34})
        self.assertLess(abs(steps.sum()*.001-535/30),.00051)

    def test_time_parameterization_preserves_ompl_corners_and_bounds(self):
        class FreeSpace:
            def valid(self,q,clearance):return True
        vertices=np.array([[0.,0.],[.12,0.],[.12,.16],[.25,.16]])
        poses,t,source=time_parameterize(vertices,['ompl_transfer']*4,FreeSpace())
        for vertex in vertices:self.assertLess(np.linalg.norm(poses-vertex,axis=1).min(),1e-12)
        self.assertLessEqual(np.abs(np.diff(poses,axis=0)*30).max(),.6+1e-10)
        self.assertLessEqual(np.abs(np.diff(poses,n=2,axis=0)*900).max(),1.2+1e-10)
        self.assertEqual(set(source),set(range(4)))

    def test_edge_bound_catches_collision_between_valid_endpoints(self):
        class PointObstacle:
            point_motion_radius=1.
            def projected_cylinder_clearance(self,q):return abs(float(q[0]))-.002,'point'
        contract=PointObstacle()
        with self.assertRaises(RuntimeError):
            certify_exclusion_edges(contract,np.array([[-.01],[.01]]),np.array([.008,.008]))
        result=certify_exclusion_edges(contract,np.array([[.05],[.06]]),np.array([.048,.058]))
        self.assertGreater(result['conservative_minimum_clearance_mm'],2.)


class SurfaceTests(unittest.TestCase):
    def test_projected_footprint_and_map_have_same_y_direction(self):
        gx=gy=np.linspace(-1,1,101)
        grid=np.zeros((101,101),bool)
        corners=np.array([[.1,.4,0],[.3,.4,0],[.3,.6,0],[.1,.6,0]])
        paint_projected_face(grid,corners,gx,gy)
        x,y=map_point(.2,.5,gx,gy,(0,0),(101,101))
        self.assertTrue(grid[::-1][y,x])
        self.assertFalse(grid[y,x])
        self.assertEqual(int(grid.sum()),11*11)

    def test_scene_pair_coverage_and_surface_reference(self):
        with tempfile.TemporaryDirectory(prefix='cavilon-contract-') as directory:
            model,geometry,site,_,_=scene(ROOT,UR5,Path(directory))
            contract=CollisionContract(model)
            self.assertEqual(len(contract.robot),len(contract.cylinder_pairs))
            self.assertIn(model.geom('tool_pad').id,contract.robot)
            self.assertIn(model.geom('tool_offset').id,contract.robot)
            self.assertTrue(any(model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH for i in contract.robot))
            self.assertGreater(model.geom_size[contract.site_id,1]*2,1.9)
            self.assertTrue(contract.valid(model.key_qpos[0]))
            independent,_=contract.projected_cylinder_clearance(model.key_qpos[0])
            self.assertGreater(independent,.003)
            targets,projection=make_targets(geometry,site)
            xy=np.array([t.position[:2]-ORIGIN[:2] for t in targets])
            radii=np.linalg.norm(xy-site[:2],axis=1)
            self.assertGreaterEqual(radii.min(),projection['projected_center_radius_mm']/1000-1e-10)


if __name__=='__main__':unittest.main()
