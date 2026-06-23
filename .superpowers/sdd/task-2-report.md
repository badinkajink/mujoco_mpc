# Task 2 Report: Gripper Collision Proxy

**Status:** DONE
**Commit:** df469e6

---

## Files Changed

### 1. Build XML (edited, then patch regenerated)
`build/mjpc/tasks/humanoid_bench/h1_2/h1_2_modified_magpie.xml`

Inside `<body name="left_magpie_gripper">`:
```xml
<geom name="left_gripper_collision" class="collision" type="box" size="0.035 0.05 0.06"
      pos="0.10 0 0" quat="0.7071068 0 0.70710685 0" contype="1" conaffinity="1"/>
```

Inside `<body name="right_magpie_gripper">`:
```xml
<geom name="right_gripper_collision" class="collision" type="box" size="0.035 0.05 0.06"
      pos="0.10 0 0" quat="0.7071068 0 0.70710685 0" contype="1" conaffinity="1"/>
```

`class="collision"` resolves from `h1_2_modified.xml` defaults to: `group=3, mass=0, density=0`.
`contype=1 conaffinity=1` makes both geoms mutually collidable and collidable with the table/floor.

### 2. Source patch (committed)
`mjpc/tasks/humanoid_bench/h1_2_modified_magpie.xml.patch`

Regenerated via:
```bash
cd build/mjpc/tasks/humanoid_bench/h1_2
diff -u h1_2_modified.xml h1_2_modified_magpie.xml > /tmp/magpie.patch.new
cp /tmp/magpie.patch.new mjpc/tasks/humanoid_bench/h1_2_modified_magpie.xml.patch
```

### 3. Twin model (edited, patch regenerated — NOT committed)
`/home/the2xman/Desktop/h12/h1_mujoco/unitree_robots/h1_2/h1_2_handless_magpie.xml`

Same geoms added but with `class="collision"` expanded inline (twin has no collision class default):
```xml
<geom name="left_gripper_collision" type="box" size="0.035 0.05 0.06"
      pos="0.10 0 0" quat="0.7071068 0 0.70710685 0" contype="1" conaffinity="1" group="3" mass="0"/>
<geom name="right_gripper_collision" type="box" size="0.035 0.05 0.06"
      pos="0.10 0 0" quat="0.7071068 0 0.70710685 0" contype="1" conaffinity="1" group="3" mass="0"/>
```

Twin patch regenerated via:
```bash
cd /home/the2xman/Desktop/h12/h1_mujoco/unitree_robots/h1_2
diff -u h1_2_handless.xml h1_2_handless_magpie.xml > h1_2_handless_magpie.xml.patch
```

---

## Static Check Output

### Run 1 (before edits — confirms FAIL):
```
left_magpie_gripper: 1 geoms, 0 collision-capable
AssertionError: left gripper has NO collision geom
```

### Run 2 (after adding geoms to build XML — confirms PASS):
```
left_magpie_gripper: 2 geoms, 1 collision-capable
right_magpie_gripper: 2 geoms, 1 collision-capable
PASS: both grippers have a collision proxy
```

### Run 3 (after `ninja copy_model_resources` reapplied patch — confirms patch survived):
```
left_magpie_gripper: 2 geoms, 1 collision-capable
right_magpie_gripper: 2 geoms, 1 collision-capable
PASS: both grippers have a collision proxy
```

---

## Concerns

None. The `ninja copy_model_resources` output showed the patch applied without rejection. The twin model (`h1_2_handless_magpie.xml`) does not have a `class="collision"` default, so the geom attributes were expanded inline (`group=3 mass=0`) to achieve the same effect — this is semantically equivalent to the MJPC model's `class="collision"` expansion.

Step 6 (twin probe) was intentionally skipped per task instructions.
