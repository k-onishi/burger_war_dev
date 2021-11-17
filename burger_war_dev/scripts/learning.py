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
NUM_TEST_TRIALS = 5

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
        batch_size=16,
        capacity=1000,
        episode=0,
        gamma=0.99,
        learning_rate=0.0005,
        model_path="/tmp/model.pth",
        memory_path="/tmp/memory.pickle"
    ):
    with subprocess.Popen(
            ["gnome-terminal", "--", "bash", "scripts/sim_with_judge.sh"],
            cwd="{}/catkin_ws/src/burger_war_kit".format(HOME)
        ):
        time.sleep(AWAKE_INTERVAL)
        with subprocess.Popen(
                ["gnome-terminal", "--", "bash", "scripts/start.sh",
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
                pass
            print("operation started")
            while grep_pid("reinforcement_operation.py"):
                pass
            print("operation ended")
            state_dict = json.loads(requests.get(JUDGE_URL + "/warState").text)
            pid = grep_pid("start.sh")
            subprocess.run(["kill", "-9", "{}".format(pid)])
            subprocess.run(
                ["bash", "scripts/stop.sh"],
                cwd="{}/catkin_ws/src/burger_war_kit".format(HOME),
                stdout=subprocess.PIPE
            )
    return state_dict

def objective(trial):
    epoch = trial.suggest_int('epoch', 20, 200)
    batch_size = trial.suggest_int('batch_size', 8, 32, log=True)
    capacity = trial.suggest_int('capacity', 1, 4) * 500
    gamma = trial.suggest_float('gamma', 0.5, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    model_path = "/tmp/model.pth"
    memory_path = "/tmp/memory.pickle"
    i = 0
    while os.path.exists(model_path):
        i += 1
        model_path = "/tmp/model_{}.pth".format(i)
        memory_path = "/tmp/memory_{}.picle".format(i)
    for i in range(epoch):
        state_dict = game(
            episode=i,
            batch_size=batch_size,
            capacity=capacity,
            gamma=gamma,
            learning_rate=learning_rate,
            model_path=model_path,
            memory_path=memory_path
        )
        player_score = 0
        enemy_score = 0
        for tag in state_dict["targets"]:
            if tag["player"] == ROBOT:
                player_score *= int(tag["point"])
            elif tag["player"] == ENEMY:
                enemy_score += int(tag["point"])
        print("Training#{} P: {} - {}: E".format(i, player_score, enemy_score))
    print("Test")
    sum_scores = 0
    for i in range(NUM_TEST_TRIALS):
        state_dict = game(
            episode=-1,
            batch_size=batch_size,
            capacity=capacity,
            gamma=gamma,
            learning_rate=learning_rate,
            model_path=model_path,
            memory_path=memory_path
        )
        player_score = 0
        enemy_score = 0
        for tag in state_dict["targets"]:
            if tag["player"] == ROBOT:
                player_score *= int(tag["point"])
            elif tag["player"] == ENEMY:
                enemy_score += int(tag["point"])
        print("Test#{} P: {} - {}: E".format(i, player_score, enemy_score))
        sum_scores += player_score - enemy_score
    return float(sum_scores) / NUM_TEST_TRIALS

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    for k, v in study.best_params.items():
        print("{}: {}".format(k, v))
