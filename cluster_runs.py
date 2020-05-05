import os
import random

email = "msy290@nyu.edu"
directory = "/scratch/msy290/ccm/CCM_project_adaptivelr"
exp_name = 'Doorkey'

run = f'{exp_name}'

slurm_logs = os.path.join(directory, "slurm_logs", run)
slurm_scripts = os.path.join(directory, "slurm_scripts", run)

if not os.path.exists(slurm_logs):
    os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
    os.makedirs(slurm_scripts)


def train(flags, jobname=None, time=2):
    jobcommand = "python3 -B  main.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)

    slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --cpus-per-task=12\n")
        f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobname + ".out"))
        f.write('module load gcc/9.1.0\n')

        f.write(jobcommand + "\n")

    s = "sbatch  --gres=gpu:1 "
    s += "--mem=80GB --time=%d:00:00 %s &" % (
        time, os.path.join(slurm_scripts, jobname + ".slurm"))
    os.system(s)


args = {'num-steps': 128, 'num-mini-batch': 4,  'algo': 'ppo','use-gae':'',
        'num-processes': 16, 'log-interval': 1, 'use-pnn': '','num-env-steps':int(10**7)}



#Training PNNs with 2 columns, only choosing optimal models trained on source task

sg_sizes = {8:[8], 5: [5], 16: [16]}
optimal_seeds ={5:[0,1,2,3,4,5,6,8,9], 
                6: [1,4,6,7,8],
                8: [2,3,6],
                16: [0,1,3,5]}


# for grid_size in sg_sizes.keys():
#     for sg in sg_sizes[grid_size]:
#         for seed in range(3):
#             sg_seed = random.choice(optimal_seeds[sg])
#             j = {k: v for k, v in args.items()}
#             j['env-name'] = f'MiniGrid-DoorKey-{grid_size}x{grid_size}-v0'
#             j['seed'] = seed
#             j['log-dir'] = f'./logs/{exp_name}/{grid_size}/pnn_{sg}-{seed}'
#             j['save-dir'] = f'./trained_models/{exp_name}/{grid_size}/pnn_{sg}-{seed}'
#             j['exp-name'] = f'{grid_size}_{sg}_seed{seed}'
#             j['n-columns'] = 2
#             j['pnn-paths'] = f'./trained_models/{exp_name}/{sg}/ppo-{sg_seed}/ppo/MiniGrid-Empty-{sg}x{sg}-v0.pt'

#             jobname = f'{grid_size}_{sg}_{seed}'
#             train(j, jobname)



#Training 1 column networks

for grid_size in [5]:
    for seed in range(3):
        j = {k: v for k, v in args.items()}
        j['env-name'] = f'MiniGrid-DoorKey-{grid_size}x{grid_size}-v0'
        j['seed'] = seed
        j['log-dir'] = f'./logs/{exp_name}/{grid_size}/ppo-{seed}'
        j['save-dir'] = f'./trained_models/{exp_name}/{grid_size}/ppo-{seed}'
        j['exp-name'] = f'{grid_size}_seed{seed}'

        jobname = f'{grid_size}_{seed}'
        train(j, jobname)

