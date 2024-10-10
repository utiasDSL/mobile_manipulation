
if __name__ == "__main__":
    # robot mdl
    from mmseq_utils import parsing
    import argparse
    import sys
    import rospy

    import mmseq_control_new.MPC as STMPC
    import mmseq_control_new.HTMPC as HTMPC
    import mmseq_control_new.HybridMPC as HybridMPC

    rospy.init_node("test_mpc")

    argv = rospy.myargv(argv=sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", required=True, help="Path to config file.")
    parser.add_argument('-n', "--name", default="", required=False, help="Name Identifier for acados ocp. This will overwrite the name in the config file if provided.")
    
    args = parser.parse_args(argv[1:])

    config = parsing.load_config(args.config)

    ctrl_config = config["controller"]
    control_class = getattr(HTMPC, ctrl_config["type"], None)
    if control_class is None:
        control_class = getattr(STMPC, ctrl_config["type"], None)
    if control_class is None:
        control_class = getattr(HybridMPC, ctrl_config["type"], None)
    controller = control_class(ctrl_config)
    
    