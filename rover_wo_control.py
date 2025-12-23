import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
 
HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene_stairs_logs.xml"
# SCENE = HERE / "scene_flat.xml"
# SCENE = HERE / "scene_rocks.xml"
# SCENE = HERE / "scene_sine.xml"
# SCENE = HERE / "scene_saw.xml"
 
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

import mujoco.viewer
viewer = mujoco.viewer.launch_passive(model, data)
 
class ConstantTorqueController:
    def __init__(self, constant_torque=0.5):
        self.constant_torque = constant_torque
        print(f"Момент на все колеса: {self.constant_torque} Н·м")
 
        self.target_speed = None
 
    def compute_wheel_torques(self):
        return np.full(6, self.constant_torque)
 
CONSTANT_TORQUE = 0.067
 
controller = ConstantTorqueController(CONSTANT_TORQUE)
 
time_steps = []
speeds = []
positions = []
applied_torques = []
 
step_count = 0
start_time = time.time()
real_start_time = start_time
 
try:
    while viewer.is_running():
        viewer.sync()
 
        current_time = time.time() - real_start_time
 
        data.ctrl[0] = 0  # diff_pitch
        data.ctrl[1] = 0  # diff_roll
 
        wheel_torques = controller.compute_wheel_torques()
 
        for i in range(6):
            data.ctrl[2 + i] = wheel_torques[i]
 
        current_speed = data.qvel[0]
        current_position = data.qpos[0]
 
        time_steps.append(current_time)
        speeds.append(current_speed)
        positions.append(current_position)
        applied_torques.append(np.mean(np.abs(wheel_torques)))  # средний момент
 
        mujoco.mj_step(model, data)
 
        if step_count % 500 == 0:
            pos_x = data.qpos[0]
            vel_x = data.qvel[0]
 
 
        step_count += 1
        time.sleep(0.001)
 
except KeyboardInterrupt:
    print("\nСимуляция прервана пользователем")
 
finally:
    viewer.close()
 
    time_steps = np.array(time_steps)
    speeds = np.array(speeds)
    positions = np.array(positions)
    applied_torques = np.array(applied_torques)
 
    if len(speeds) > 0:
        total_time = time_steps[-1]
        total_distance = positions[-1] - positions[0]
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        min_speed = np.min(speeds)
        speed_std = np.std(speeds)
 
        controller.target_speed = avg_speed
 
        squared_deviations = (speeds - controller.target_speed) ** 2
        rms_deviation = np.sqrt(np.mean(squared_deviations))
 
        target_range_min = controller.target_speed * 0.9
        target_range_max = controller.target_speed * 1.1
        in_target_range = np.sum((speeds >= target_range_min) & (speeds <= target_range_max))
        percent_in_target = (in_target_range / len(speeds)) * 100
 
        print(f"Всего шагов: {step_count}")
        print(f"Общее время: {total_time:.2f} сек")
        print(f"Финальная позиция: X={positions[-1]:.2f}м")
        print(f"Финальная скорость: {speeds[-1]:.2f} м/с")
        print(f"Средняя скорость (целевая): {avg_speed:.3f} ± {speed_std:.3f} м/с")
        print(f"Максимальная скорость: {max_speed:.3f} м/с")
        print(f"Минимальная скорость: {min_speed:.3f} м/с")
        print(f"Пройденное расстояние: {total_distance:.2f} м")
        print(f"Приложенный момент: {controller.constant_torque} Н·м на все колеса")
 
        print(f"\nСРЕДНЕКВАДРАТИЧНОЕ ОТКЛОНЕНИЕ (RMS): {rms_deviation:.4f} м/с")
        print(f"Время в целевом диапазоне (±10% от средней): {percent_in_target:.1f}%")
 
    if len(speeds) > 0:
 
        plt.figure(figsize=(14, 8))
 
        plt.subplot(2, 2, 1)
        plt.plot(time_steps, speeds, 'r-', linewidth=1.5, label='Скорость ровера')
        plt.axhline(y=controller.target_speed, color='b', linestyle='--',
                   linewidth=2, label=f'Целевая: {controller.target_speed:.3f} м/с')
        plt.fill_between(time_steps,
                        controller.target_speed * 0.9,
                        controller.target_speed * 1.1,
                        alpha=0.2, color='green', label='Целевой диапазон ±10%')
        plt.xlabel('Время (сек)')
        plt.ylabel('Скорость (м/с)')
        plt.title(f'Скорость ровера (момент: {controller.constant_torque} Н·м)')
        plt.grid(True, alpha=0.3)
        plt.legend()
 
        plt.subplot(2, 2, 2)
        plt.plot(time_steps, positions, 'g-', linewidth=2)
        plt.xlabel('Время (сек)')
        plt.ylabel('Позиция X (м)')
        plt.title('Позиция ровера во времени')
        plt.grid(True, alpha=0.3)
 
        plt.subplot(2, 2, 3)
        plt.hist(speeds, bins=30, edgecolor='black', alpha=0.7, color='salmon')
        plt.axvline(x=controller.target_speed, color='b', linestyle='--',
                   linewidth=2, label=f'Целевая: {controller.target_speed:.3f} м/с')
        plt.axvline(x=avg_speed, color='r', linestyle='-',
                   linewidth=2, label=f'Средняя: {avg_speed:.3f} м/с')
        plt.xlabel('Скорость (м/с)')
        plt.ylabel('Частота')
        plt.title('Распределение скоростей')
        plt.grid(True, alpha=0.3)
        plt.legend()
 
        plt.subplot(2, 2, 4)
 
        median_speed = np.median(speeds)
        low_speed_threshold = controller.target_speed * 0.9
        high_speed_threshold = controller.target_speed * 1.1
 
        low_speed_mask = speeds < low_speed_threshold
        high_speed_mask = speeds > high_speed_threshold
 
        plt.plot(time_steps, speeds, 'r-', linewidth=1.5, label='Скорость')
        plt.axhline(y=controller.target_speed, color='b', linestyle='--',
                   linewidth=2, label=f'Целевая: {controller.target_speed:.3f} м/с')
 
        plt.fill_between(time_steps, 0, speeds,
                        where=low_speed_mask,
                        color='red', alpha=0.3, label=f'Ниже целевого (<{low_speed_threshold:.3f})')
 
        plt.fill_between(time_steps, 0, speeds,
                        where=high_speed_mask,
                        color='green', alpha=0.3, label=f'Выше целевого (>{high_speed_threshold:.3f})')
 
        plt.xlabel('Время (сек)')
        plt.ylabel('Скорость (м/с)')
        plt.title('Скорость с выделением отклонений от целевого диапазона')
        plt.grid(True, alpha=0.3)
        plt.legend()
 
        plt.suptitle(f'СИМУЛЯЦИЯ С ПОСТОЯННЫМ МОМЕНТОМ ({controller.constant_torque} Н·м)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
 
        plt.figure(figsize=(12, 6))
 
        plt.plot(time_steps, speeds, 'r-', linewidth=2, label='Скорость ровера', alpha=0.8)
        plt.axhline(y=controller.target_speed, color='b', linestyle='--',
                   linewidth=2.5, label=f'Целевая скорость: {controller.target_speed:.3f} м/с')
 
        if len(speeds) > 10:
            window_size = min(100, len(speeds) // 10)
            smoothed_speed = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid')
            smoothed_time = time_steps[window_size-1:]
 
            plt.plot(smoothed_time, smoothed_speed, 'g-', linewidth=1.5,
                    label=f'Скользящее среднее (окно {window_size} точек)', alpha=0.7)
 
        plt.axhline(y=avg_speed, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Среднее: {avg_speed:.3f} м/с')
        plt.axhline(y=median_speed, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Медиана: {median_speed:.3f} м/с')
 
        plt.xlabel('Время (сек)', fontsize=12)
        plt.ylabel('Скорость (м/с)', fontsize=12)
        plt.title(f'Анализ скорости ровера (постоянный момент: {controller.constant_torque} Н·м)',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
 
        stats_text = (f'Приложенный момент: {controller.constant_torque} Н·м\n'
                     f'Целевая скорость: {controller.target_speed:.3f} м/с\n'
                     f'Средняя скорость: {avg_speed:.3f} м/с\n'
                     f'Медиана скорости: {median_speed:.3f} м/с\n'
                     f'Станд. отклонение: {speed_std:.3f} м/с\n'
                     f'RMS отклонение: {rms_deviation:.3f} м/с\n'
                     f'Максимальная: {max_speed:.3f} м/с\n'
                     f'Минимальная: {min_speed:.3f} м/с\n'
                     f'Пройдено: {total_distance:.2f} м\n'
                     f'В целевом диапазоне: {percent_in_target:.1f}%')
 
        plt.text(0.98, 0.98, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
 
        plt.tight_layout()
        plt.show()
 