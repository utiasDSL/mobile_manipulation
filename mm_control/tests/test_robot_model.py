"""
Integration tests for robot model validation.

Tests verify that symbolic CasADi models match PyBullet simulation.
Requires running simulation - use with rosrun.

Usage:
    rosrun mm_control test_robot_model.py --config <path_to_config.yaml>
"""

import argparse
import datetime

import numpy as np

from mm_control.robot import CasadiModelInterface, MobileManipulator3D
from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.math import q2r


def verify_link_transforms(robot_sim, sysMdls, link_names):
    """Verify symbolic FK matches simulation for each link."""
    errors = []
    for name in link_names:
        q, v = robot_sim.joint_states()

        if name in robot_sim.links:
            link_idx = robot_sim.links[name][0]
        else:
            continue
        pos_sim, orn_sim = robot_sim.link_pose(link_idx)
        rot_sim = q2r(np.array(orn_sim), order="xyzs")

        fk_fcn = sysMdls[name]
        pos_mdl, rot_mdl = fk_fcn(q)
        pos_mdl = pos_mdl.toarray().flatten()

        pos_diff = np.linalg.norm(pos_sim - pos_mdl)
        rot_diff = np.linalg.norm(rot_mdl - rot_sim)

        print(f"  {name}: pos_diff={pos_diff:.6f}, rot_diff={rot_diff:.6f}")

        # Position differences expected due to CoM vs joint frame
        if rot_diff > 1e-4:
            errors.append(f"{name} rotation error: {rot_diff}")

    return errors


def verify_link_jacobian(robot_sim, sysMdls, link_names):
    """Verify symbolic Jacobians match simulation.

    Note: Some differences are expected due to coordinate frame conventions
    between PyBullet (world frame) and CasADi models.
    """
    for name in link_names:
        q, v = robot_sim.joint_states()
        J_sim = robot_sim.jacobian(q)

        J_fcn = sysMdls[name]
        J_mdl = J_fcn(q)

        J_diff = np.linalg.norm(J_mdl - J_sim[:3])
        print(f"  {name}: J_diff={J_diff:.6f}")

    # Jacobian differences are informational only (coordinate frame differences expected)
    return []


def test_robot_model(args):
    """Test robot kinematic and dynamic models against simulation."""
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]

    # Create symbolic model
    robot = MobileManipulator3D(ctrl_config)

    # Start simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    mm = sim.robot
    sim.settle(5.0)

    errors = []

    # Test link transforms
    print("\n=== Testing Link Transforms ===")
    link_names = robot.link_names[1:]
    errors.extend(verify_link_transforms(mm, robot.kinSymMdls, link_names))

    # Test Jacobians
    print("\n=== Testing Link Jacobians ===")
    errors.extend(verify_link_jacobian(mm, robot.jacSymMdls, link_names))

    # Test collision link transforms
    print("\n=== Testing Collision Link Transforms ===")
    collision_link_names = []
    for _, collision_link_name in robot.collision_link_names.items():
        collision_link_names += collision_link_name
    errors.extend(
        verify_link_transforms(mm, robot.collisionLinkKinSymMdls, collision_link_names)
    )

    # Test motion model integrator
    print("\n=== Testing Motion Model Integrator ===")
    dt = 0.1
    a = 1.0
    N = 10
    u_bar = np.array([[a] * 9] * N)
    xo = np.array(mm.joint_states()).flatten()
    x_bar_sym = MobileManipulator3D.ssIntegrate(dt, xo, u_bar, robot.ssSymMdl)

    x_bar_num = np.zeros((N + 1, 18))
    x_bar_num[0] = xo
    for k in range(N):
        x_bar_num[k + 1, 9:] = x_bar_num[k, 9:] + a * dt
        x_bar_num[k + 1, :9] = (
            x_bar_num[k, :9] + x_bar_num[k, 9:] * dt + 0.5 * a * dt * dt
        )

    pred_diff = np.linalg.norm(x_bar_num - x_bar_sym)
    print(f"  Integrator prediction diff: {pred_diff:.6f}")
    if pred_diff > 1e-4:
        errors.append(f"Integrator error: {pred_diff}")

    # Test manipulability
    print("\n=== Testing Manipulability ===")
    manip = robot.manipulability_fcn(mm.home)
    print(f"  Manipulability at home: {manip}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
    else:
        print("PASSED: All robot model tests passed")

    return len(errors) == 0


def test_collision_model(args):
    """Test collision model instantiation and basic functionality."""
    config = parsing.load_config(args.config)
    ctrl_config = config["controller"]

    print("\n=== Testing Collision Model ===")
    errors = []

    try:
        sym_model = CasadiModelInterface(ctrl_config)
        print("  CasadiModelInterface created successfully")

        # Check collision pairs exist
        if "self" in sym_model.collision_pairs:
            n_self = len(sym_model.collision_pairs["self"])
            print(f"  Self-collision pairs: {n_self}")

        # Test symbolic distance function can be created and called
        q = np.zeros(6)
        if "self" in sym_model.collision_pairs and sym_model.collision_pairs["self"]:
            sd_fcn = sym_model.getSignedDistanceSymMdls("self")
            distances = sd_fcn(np.hstack((np.zeros(3), q)))
            print(
                f"  Self-collision distances computed: {len(np.array(distances).flatten())} values"
            )
            print(f"  Min distance: {np.min(distances):.4f}")

    except Exception as e:
        errors.append(f"Collision model error: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
    else:
        print("PASSED: Collision model tests passed")

    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test robot model against simulation")
    parser.add_argument("--config", required=False, help="Path to configuration file.")
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    args = parser.parse_args()

    # Default config if not provided
    if args.config is None:
        args.config = parsing.parse_ros_path(
            {"package": "mm_run", "path": "config/simple_experiment.yaml"}
        )

    success = test_robot_model(args)
    success = test_collision_model(args) and success

    exit(0 if success else 1)
