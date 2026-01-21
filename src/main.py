import argparse
from pathlib import Path

from sim.sim_runner import run_simulation
from control import rover_w_control, rover_wo_control


HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "assets" / "scene_stairs_logs.xml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=str, default=str(DEFAULT_SCENE))
    p.add_argument("--mode", choices=["w", "wo"], default="w")
    p.add_argument("--torque", type=float, default=0.067)
    args = p.parse_args()

    if args.mode == "w":
        controller_factory = rover_w_control.create_controller
        analyze = rover_w_control.analyze_and_plot
    else:
        controller_factory = lambda model, data: rover_wo_control.create_controller(model, data, torque=args.torque)
        analyze = rover_wo_control.analyze_and_plot

    logs = run_simulation(args.scene, controller_factory)
    analyze(logs)


if __name__ == "__main__":
    main()
