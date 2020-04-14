import os

exp_name = 'gridworld_new'

write_to_file = ''


def create_command(flags):
    global write_to_file

    jobcommand = "python3 -B  main.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)

    write_to_file += jobcommand + ' & \n'


args = {'num-steps': 128, 'num-mini-batch': 4, 'lr': 2.5e-4, 'algo': 'ppo',
        'num-processes': 12, 'log-interval': 1, 'use-linear-lr-decay': '',
        'use-pnn': '', 'num-env-steps': int(1E6)}

grid_size = 8
filename='ppo_run.sh'

for seed in [0, 1, 2]:
    j = {k: v for k, v in args.items()}
    j['n-columns'] = 2
    j['env-name'] = f'MiniGrid-Empty-{grid_size}x{grid_size}-v0'
    j['seed'] = seed
    j['log-dir'] = f'./logs/{exp_name}/{grid_size}/pnn_5_{seed}'
    j['save-dir'] = f'./trained_models/{exp_name}/{grid_size}/pnn_5_{seed}'
    j['exp-name'] = f'{grid_size}_5_seed{seed}'
    j['pnn-paths'] = f'trained_models/gridworld/5/ppo/MiniGrid-Empty-5x5-v0.pt'

    create_command(j)

with open(filename, 'w') as f:
    f.write(write_to_file)
