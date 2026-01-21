import numpy as np
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, target, current):
        error = target - current

        p = self.kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10.0, 10.0)
        i = self.ki * self.integral

        derivative = (error - self.prev_error) / self.dt
        d = self.kd * derivative
        self.prev_error = error

        output = p + i + d
        output = np.clip(output, -4.0, 4.0)

        return output


class RoverController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.speed_pid = PIDController(kp=10.5, ki=3, kd=0.01, dt=0.002)
        self.target_speed = 0.3

    def get_wheel_contacts(self):
        contacts = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            for wheel_idx in range(6):
                wheel_geom_id = 6 + wheel_idx
                if contact.geom1 == wheel_geom_id or contact.geom2 == wheel_geom_id:
                    contacts[wheel_idx] = 1

        return contacts

    def compute_wheel_torques(self):
        current_speed = self.data.qvel[0]
        base_torque = self.speed_pid.compute(self.target_speed, current_speed)

        contacts = self.get_wheel_contacts()
        torques = np.zeros(6)

        wheels_on_ground = np.sum(contacts)

        if wheels_on_ground > 0:
            torque_per_wheel = base_torque / wheels_on_ground

            for i in range(6):
                if contacts[i] == 1:  # колесо на земле
                    torques[i] = torque_per_wheel
                else:  # колесо в воздухе
                    torques[i] = 0.1 * base_torque
        else:
            torques.fill(0.1 * base_torque)

        return torques


def create_controller(model, data):
    controller = RoverController(model, data)
    print(f"Целевая скорость: {controller.target_speed} м/с")
    return controller


def analyze_and_plot(logs):
    # Берём данные из runner'а
    controller = logs["controller"]
    time_steps = np.array(logs["time_steps"])
    speeds = np.array(logs["speeds"])
    positions = np.array(logs["positions"])
    step_count = logs["step_count"]

    if len(speeds) == 0:
        print("Нет данных для анализа.")
        return

    target_speeds = np.full_like(speeds, controller.target_speed, dtype=float)

    print(f"Всего шагов: {step_count}")
    print(f"Общее время: {time_steps[-1]:.2f} сек")
    print(f"Финальная позиция: X={positions[-1]:.2f}м")
    print(f"Финальная скорость: {speeds[-1]:.2f} м/с")
    print(f"Средняя скорость: {np.mean(speeds):.3f} ± {np.std(speeds):.3f} м/с")
    print(f"Максимальная скорость: {np.max(speeds):.3f} м/с")
    print(f"Минимальная скорость: {np.min(speeds):.3f} м/с")

    squared_deviations = (speeds - controller.target_speed) ** 2
    rms_deviation = np.sqrt(np.mean(squared_deviations))
    print(f"\nСРЕДНЕКВАДРАТИЧНОЕ ОТКЛОНЕНИЕ (RMS): {rms_deviation:.4f} м/с")

    target_range_min = controller.target_speed * 0.9
    target_range_max = controller.target_speed * 1.1
    in_target_range = np.sum((speeds >= target_range_min) & (speeds <= target_range_max))
    percent_in_target = (in_target_range / len(speeds)) * 100
    print(f"Время в целевом диапазоне (±10%): {percent_in_target:.1f}%")

    total_distance = positions[-1] - positions[0]
    print(f"Пройденное расстояние: {total_distance:.2f} м")

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time_steps, speeds, "b-", linewidth=1.5, label="Скорость ровера")
    plt.plot(time_steps, target_speeds, "r--", linewidth=2, label="Целевая скорость")
    plt.fill_between(
        time_steps,
        controller.target_speed * 0.9,
        controller.target_speed * 1.1,
        alpha=0.2,
        color="green",
        label="Целевой диапазон ±10%",
    )
    plt.xlabel("Время (сек)")
    plt.ylabel("Скорость (м/с)")
    plt.title("Скорость ровера во времени")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time_steps, positions, "g-", linewidth=2)
    plt.xlabel("Время (сек)")
    plt.ylabel("Позиция X (м)")
    plt.title("Позиция ровера во времени")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(speeds, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
    plt.axvline(
        x=controller.target_speed,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Цель: {controller.target_speed} м/с",
    )
    plt.axvline(
        x=np.mean(speeds),
        color="b",
        linestyle="-",
        linewidth=2,
        label=f"Средняя: {np.mean(speeds):.3f} м/с",
    )
    plt.xlabel("Скорость (м/с)")
    plt.ylabel("Частота")
    plt.title("Распределение скоростей")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    low_speed_mask = speeds < (controller.target_speed * 0.5)
    high_speed_mask = speeds > (controller.target_speed * 1.2)

    plt.plot(time_steps, speeds, "b-", linewidth=1.5, label="Скорость")
    plt.plot(time_steps, target_speeds, "r--", linewidth=2, label="Цель")

    if np.any(low_speed_mask):
        plt.fill_between(
            time_steps,
            0,
            speeds,
            where=low_speed_mask,
            color="red",
            alpha=0.3,
            label="Низкая скорость (<50% цели)",
        )

    if np.any(high_speed_mask):
        plt.fill_between(
            time_steps,
            0,
            speeds,
            where=high_speed_mask,
            color="green",
            alpha=0.3,
            label="Высокая скорость (>120% цели)",
        )

    plt.xlabel("Время (сек)")
    plt.ylabel("Скорость (м/с)")
    plt.title("Скорость с выделением аномалий")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, speeds, "b-", linewidth=2, label="Скорость ровера", alpha=0.8)
    plt.plot(
        time_steps,
        target_speeds,
        "r--",
        linewidth=2.5,
        label=f"Целевая скорость ({controller.target_speed} м/с)",
    )

    if len(speeds) > 10:
        window_size = min(100, len(speeds) // 10)
        smoothed_speed = np.convolve(speeds, np.ones(window_size) / window_size, mode="valid")
        smoothed_time = time_steps[window_size - 1 :]
        plt.plot(
            smoothed_time,
            smoothed_speed,
            "g-",
            linewidth=1.5,
            label=f"Скользящее среднее (окно {window_size} точек)",
            alpha=0.7,
        )

    plt.xlabel("Время (сек)", fontsize=12)
    plt.ylabel("Скорость (м/с)", fontsize=12)
    plt.title("Анализ скорости ровера на всей траектории", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    stats_text = (
        f"Средняя: {np.mean(speeds):.3f} м/с\n"
        f"Медиана: {np.median(speeds):.3f} м/с\n"
        f"Станд. отклонение: {np.std(speeds):.3f} м/с\n"
        f"Максимум: {np.max(speeds):.3f} м/с\n"
        f"Минимум: {np.min(speeds):.3f} м/с\n"
        f"RMS отклонение: {rms_deviation:.3f} м/с"
    )

    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()
