import json
import os
import requests
import subprocess
import time


HOME = os.environ['HOME']
AWAKE_INTERVAL = 20
JUDGE_URL = "http://127.0.0.1:5000"
ROBOT = "r"
ENEMY = "b"

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
            ["gnome-terminal", "--", "bash", "scripts/sim_with_judge.sh",
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
        time.sleep(AWAKE_INTERVAL)
        with subprocess.Popen(
                ["gnome-terminal", "--", "bash", "scripts/start.sh"],
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

if __name__ == "__main__":
    for i in range(2):
        state_dict = game(episode=i)
        player_score = 0
        enemy_score = 0
        for tag in state_dict["targets"]:
            if tg["player"] == ROBOT:
                player_score = int(tag["point"])
            elif tg["player"] == ENEMY:
                enemy_score = int(tag["point"])
        print("P: {} - {}: E".format(player_score, enemy_score))
