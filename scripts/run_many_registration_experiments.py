import argparse
import subprocess

from stucco.env import poke
from stucco import cfg

parser = argparse.ArgumentParser(description='Run many registration poking experiments')
parser.add_argument('--experiment',
                    choices=['build', 'baseline', 'poke'],
                    default='poke',
                    help='which experiment to run')
parser.add_argument('--registration', nargs='+',
                    default=['volumetric'],
                    help='which registration methods to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
task_map = {level.name.lower(): level for level in poke.Levels}
parser.add_argument('--task', type=str, nargs='+',
                    default=['all'],
                    choices=['all'] + list(task_map.keys()), help='what tasks to run')
parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
parser.add_argument('--dry', action='store_true', help='print the commands to run without execution')

args = parser.parse_args()

if __name__ == "__main__":
    if 'all' in args.task:
        args.task = task_map.keys()
    for registration in args.registration:
        for task in args.task:
            to_run = ["python", f"{cfg.ROOT_DIR}/scripts/registration_experiments.py", args.experiment,
                      "--registration", registration, "--task", task, "--seed"] + [str(s) for s in args.seed]
            if args.name is not None:
                to_run.append("--name")
                to_run.append(args.name)
            if args.no_gui:
                to_run.append("--no_gui")

            print(" ".join(to_run))
            if not args.dry:
                subprocess.run(to_run)