# Simrobots_project
to be done
# Rover_project — симуляция марсохода в MuJoCo

## Цель и задачи
Цель проекта — смоделировать движение марсохода в физическом движке MuJoCo и сравнить два подхода управления:
- управление скоростью (PID-регулятор)
- движение с постоянным моментом на колёсах

Задачи:
- запуск симуляции на разных сценах
- реализация двух стратегий управления
- сбор и вывод метрик + построение графиков

---

## Структура проекта
- `src/main.py` — единая точка входа (CLI)
- `src/sim/sim_runner.py` — общий цикл симуляции (MuJoCo + viewer + логи)
- `src/control/rover_w_control.py` — PID-управление + анализ и графики
- `src/control/rover_wo_control.py` — постоянный момент + анализ и графики
- `src/assets/*.xml` — модель `rover.xml` и сцены `scene_*.xml`

---

## Установка и запуск

### Установка зависимостей
pip install -r requirements.txt

### Запуск 
python src/main.py
По умолчанию запускается управление PID на сцене scene_stairs_logs.xml

python src/main.py --mode w --scene src/assets/scene_stairs_logs.xml
"--mode" - выбор режима управления w/wo (with/without) PID
"--scene" - путь к .xml файлу с сценой
"--torque" - момент на колесах (в режиме without PID)

Филиппов А.В.; Тихонов В.С.; Фомин В.М.