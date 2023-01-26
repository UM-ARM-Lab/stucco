import argparse
import subprocess

from stucco.env import poke
from stucco import cfg
import logging
import os
from datetime import datetime

logger = logging.getLogger(__file__)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}_run_many.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])


parser = argparse.ArgumentParser(description='Run many registration poking experiments')
parser.add_argument('--experiment',
                    choices=['build', 'baseline', 'poke', 'generate-plausible-set', 'evaluate-plausible-diversity',
                             'plot-plausible-set', 'plot-poke-ce', 'plot-poke-pd',
                             'debug'],
                    default='poke',
                    help='which experiment to run')
parser.add_argument('--registration', nargs='+',
                    default=['volumetric'],
                    help='which registration methods to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
parser.add_argument('--read_stored', action='store_true', help='read stored output instead of rerunning when possible')
task_map = {level.name.lower(): level for level in poke.Levels}
parser.add_argument('--task', type=str, nargs='+',
                    default=['all'],
                    choices=['all'] + list(task_map.keys()), help='what tasks to run')
parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
parser.add_argument('--dry', action='store_true', help='print the commands to run without execution')
parser.add_argument('rest', nargs='*', help='arguments to forward; should be separated from other arguments with --')

args = parser.parse_args()

runs = {}
if __name__ == "__main__":
    if 'all' in args.task:
        args.task = task_map.keys()
    for registration in args.registration:
        for task in args.task:
            to_run = ["python", f"{cfg.ROOT_DIR}/scripts/registration_experiments.py", args.experiment,
                      "--registration", registration, "--task", task, "--seed"] + [str(s) for s in
                                                                                   args.seed] + args.rest
            if args.name is not None:
                to_run.append("--name")
                to_run.append(args.name)
            if args.no_gui:
                to_run.append("--no_gui")
            if args.read_stored:
                to_run.append("--read_stored")

            cmd = " ".join(to_run)
            logger.info(cmd)
            if not args.dry:
                completed = subprocess.run(to_run)
                runs[cmd] = completed.returncode
# log runs
logger.info("\n\n\n")
for cmd, status in runs.items():
    if status != 0:
        logger.info(f"FAILED: {cmd}")
logger.info(f"{len(runs)} command run, {len([status for status in runs.values() if status != 0])} failures\n\n\n")
