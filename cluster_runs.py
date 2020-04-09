import os
import random

email = "msy290@nyu.edu"
directory = "/misc/kcgscratch1/ChoGroup/srikar/ccm/CCM_project"
exp = 'pnn_runs'

run = f'{exp}'

slurm_logs = os.path.join(directory, "slurm_logs", run)
slurm_scripts = os.path.join(directory, "slurm_scripts", run)

if not os.path.exists(slurm_logs):
    os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
    os.makedirs(slurm_scripts)


def train(flags, jobname=None, time=24):
    jobcommand = "srun python3 -B  main.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)

    slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name" + "=" + jobname + "\n")
        f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobname + ".out"))
        f.write("#SBATCH --qos=batch\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        f.write("source /misc/kcgscratch1/ChoGroup/srikar/db_env/bin/activate\n")
        f.write("module load cuda-10.0\n")
        f.write('module load gcc-8.2\n')
        f.write("module load gcc-9.1\n")
        f.write('export NCCL_DEBUG=INFO\n')
        f.write('export MASTER_PORT=$((12000 + RANDOM % 20000))\n')
        f.write('export NCCL_DEBUG=INFO\n')
        f.write('export PYTHONFAULTHANDLER=1\n')

        f.write(jobcommand + "\n")

    s = "sbatch --qos batch --gres=gpu:1 --constraint='gpu_12gb' --nodes=1 "
    s += "--mem=60GB --time=%d:00:00 %s &" % (
        time, os.path.join(slurm_scripts, jobname + ".slurm"))
    os.system(s)


job = {
    "algo": 'ppo', "num-processes": 64, "num-steps": 128,
    'num-mini-batch': 4, 'log-interval': 1,
    'use-linear-lr-decay': '', 'use-gae': '', 'n-columns': 2,
    'clip-param': 0.1, 'value-loss-coef': 0.5, 'use-pnn': '',
}

time = 48

target_env = 'BreakoutNoFrameskip-v4'
source_envs_list = ['Pong', 'Seaquest']

for source_env in source_envs_list:

    for seed in [0, 1, 2, 3]:

        j = {k: v for k, v in job.items()}

        j['env-name'] = target_env

        if source_env == 'both':
            j['n-columns'] = 3
            j['pnn-paths'] = f'trained_models/Pong/{seed}/ppo/PongNoFrameskip-v4.pt ' + \
                             f'trained_models/Seaquest/{seed}/ppo/SeaquestNoFrameskip-v4.pt '

        else:
            j['pnn-paths'] = f'trained_models/{source_env}/{seed}/ppo/{source_env}NoFrameskip-v4.pt '

        j['seed'] = seed

        jobname = f'pnn_{source_env}_{seed}'
        j['log-dir'] = f'logs/Breakout/pnn_{source_env}/{seed}'
        j['save-dir'] = f'trained_models/Breakout/pnn_{source_env}/{seed}'

        train(j, jobname=jobname, time=time)

