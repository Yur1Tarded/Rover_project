import time
from pathlib import Path

import mujoco
import mujoco.viewer


def run_simulation(scene_path, controller_factory, *, sleep=0.001):

    scene_path = Path(scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    controller = controller_factory(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)

    time_steps = []
    speeds = []
    positions = []
    step_count = 0
    start_time = time.time()

    try:
        while viewer.is_running():
            viewer.sync()

            data.ctrl[0] = 0  # diff_pitch
            data.ctrl[1] = 0  # diff_roll

            wheel_torques = controller.compute_wheel_torques()
            for i in range(6):
                data.ctrl[2 + i] = float(wheel_torques[i])

            current_time = time.time() - start_time
            time_steps.append(current_time)
            speeds.append(float(data.qvel[0]))
            positions.append(float(data.qpos[0]))

            mujoco.mj_step(model, data)

            step_count += 1
            time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nСимуляция прервана пользователем")
    finally:
        viewer.close()

    return {
        "time_steps": time_steps,
        "speeds": speeds,
        "positions": positions,
        "step_count": step_count,
        "controller": controller,
    }
