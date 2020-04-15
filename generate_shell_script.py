import os

exp_name = 'gridworld'

write_to_file = ''


def create_command(flags):
    global write_to_file

    jobcommand = "python3 -B  main.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)

    write_to_file += jobcommand + ' \n'


args = {'num-steps': 128, 'num-mini-batch': 4, 'lr': 2.5e-4, 'algo': 'ppo',
        'num-processes': 20, 'log-interval': 1, 'use-linear-lr-decay': '',
        'use-pnn': '', 'num-env-steps': int(5E6)}

grid_size = 16
filename='ppo_run.sh'

for seed in [0, 1, 2]:
    j = {k: v for k, v in args.items()}
    j['env-name'] = f'MiniGrid-Empty-{grid_size}x{grid_size}-v0'
    j['seed'] = seed
    j['log-dir'] = f'./logs/{exp_name}/{grid_size}/{seed}'
    j['save-dir'] = f'./trained_models/{exp_name}/{grid_size}/{seed}'
    j['exp-name'] = f'{grid_size}_seed{seed}'


    create_command(j)


# for sg in [5]:
#     for seed in [0, 1, 2]:
#         j = {k: v for k, v in args.items()}
#         j['n-columns'] = 2
#         j['env-name'] = f'MiniGrid-Empty-{grid_size}x{grid_size}-v0'
#         j['seed'] = seed
#         j['log-dir'] = f'./logs/{exp_name}/{grid_size}/pnn_{sg}_{seed}'
#         j['save-dir'] = f'./trained_models/{exp_name}/{grid_size}/pnn_{sg}_{seed}'
#         j['exp-name'] = f'{grid_size}_{sg}_seed{seed}'
#         if sg == 8:
#             j['pnn-paths'] = f'trained_models/gridworld_new/{sg}/{seed}/ppo/MiniGrid-Empty-{sg}x{sg}-v0.pt'
#         else:
#             j['pnn-paths'] = f'trained_models/gridworld_new/{sg}/1/ppo/MiniGrid-Empty-{sg}x{sg}-v0.pt'

#         create_command(j)

with open(filename, 'w') as f:
    f.write(write_to_file)
