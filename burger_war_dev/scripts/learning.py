import json
import os
import requests
import subprocess
import time

import optuna


HOME = os.environ['HOME']
AWAKE_INTERVAL = 20
JUDGE_URL = "http://127.0.0.1:5000"
ROBOT = "r"
ENEMY = "b"
GREP_INTERVAL = 1
NUM_TEST_TRIALS = 5
WAIT_BURGER_WORLD = 20

def grep_pid(cmd):
    with subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE) as ps:
        with subprocess.Popen(
            ["grep"] + list(cmd.split()),
            encoding='utf-8',
            stdin=ps.stdout,
            stdout=subprocess.PIPE
        ) as grep:
            with subprocess.Popen(
                ["grep", "-v", "grep"],
                encoding='utf-8',
                stdin=grep.stdout,
                stdout=subprocess.PIPE
            ) as vgrep:
                result = list(vgrep.stdout.read().split())
                if len(result) > 0:
                    return result[1]
                else:
                    return None

def game(
        enemy_level=1,
        batch_size=16,
        capacity=1000,
        episode=0,
        gamma=0.99,
        learning_rate=0.0005,
        model_path="/tmp/model.pth",
        memory_path="/tmp/memory.pickle"
    ):
    def stop():
        pid = grep_pid("start.sh")
        subprocess.run(["kill", "-9", "{}".format(pid)])
        subprocess.run(
            ["bash", "scripts/stop.sh"],
            cwd="{}/catkin_ws/src/burger_war_kit".format(HOME),
            stdout=subprocess.PIPE
        )
    with subprocess.Popen(
            ["gnome-terminal", "--", "bash", "scripts/sim_with_judge.sh"],
            cwd="{}/catkin_ws/src/burger_war_kit".format(HOME)
        ):
        time.sleep(AWAKE_INTERVAL)
        with subprocess.Popen(
                ["gnome-terminal", "--", "bash", "scripts/start.sh",
                    "-l", "{}".format(enemy_level),
                    "-b", "{}".format(batch_size),
                    "-c", "{}".format(capacity),
                    "-e", "{}".format(episode),
                    "-g", "{}".format(gamma),
                    "-r", "{}".format(learning_rate),
                    "-m", "{}".format(model_path),
                    "-p", "{}".format(memory_path),
                ],
                cwd="{}/catkin_ws/src/burger_war_kit".format(HOME)
            ):
            while not grep_pid("reinforcement_operation.py"):
                time.sleep(GREP_INTERVAL)
            left_time = WAIT_BURGER_WORLD
            while grep_pid("reinforcement_operation.py"):
                if grep_pid("burger_field.world") is None:
                    if left_time > 0:
                        left_time -= 1
                    else:
                        pid = grep_pid("reinforcement_operation.py")
                        subprocess.run(["kill", "-9", "{}".format(pid)])
                        stop()
                        print("burger_field was dead")
                        return None
                time.sleep(GREP_INTERVAL)
            state_dict = json.loads(requests.get(JUDGE_URL + "/warState").text)
            stop()
    return state_dict

def get_score(state_dict):
    score = state_dict["scores"]
    return score[ROBOT], score[ENEMY]

def objective(trial):
    epoch = trial.suggest_int('epoch', 20, 200)
    batch_size = trial.suggest_int('batch_size', 8, 32, log=True)
    capacity = trial.suggest_int('capacity', 1, 4) * 500
    gamma = trial.suggest_float('gamma', 0.5, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    enemy_level = trial.suggest_categorical('enemy_level', list(range(1, 12)))
    model_path = "/tmp/model.pth"
    memory_path = "/tmp/memory.pickle"
    i = 0
    if os.path.exists(model_path):
        os.remove(model_path)
        os.remove(memory_path)
    for i in range(epoch):
        state_dict = None
        while state_dict is None:
            state_dict = game(
                episode=i,
                batch_size=batch_size,
                capacity=capacity,
                gamma=gamma,
                learning_rate=learning_rate,
                model_path=model_path,
                memory_path=memory_path
            )
        player_score, enemy_score = get_score(state_dict)
        print("Training#{} P: {} - {}: E".format(i, player_score, enemy_score))
    print("Test")
    sum_scores = 0
    for i in range(NUM_TEST_TRIALS):
        state_dict = None
        while state_dict is None:
            state_dict = game(
                enemy_level=11,
                episode=-1,
                batch_size=batch_size,
                capacity=capacity,
                gamma=gamma,
                learning_rate=learning_rate,
                model_path=model_path,
                memory_path=memory_path
            )
        player_score, enemy_score = get_score(state_dict)
        print("Test#{} P: {} - {}: E".format(i, player_score, enemy_score))
        sum_scores += player_score - enemy_score
    return float(sum_scores) / NUM_TEST_TRIALS

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    for k, v in study.best_params.items():
        print("{}: {}".format(k, v))
