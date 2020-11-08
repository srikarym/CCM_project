# pytorch-a2c-ppo-acktr


Link to original repository - [Link](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)


## Requirements

* Python 3 (t)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```


## Training

#### PPO Single column

```bash
python main.py --env-name "PongNoFrameskip-v4" --use-pnn --use-gae   --num-processes 8 --num-steps 128 --num-mini-batch 4  --use-linear-lr-decay 
```

#### Progressive neural network with 2 columns

```bash
python main.py --env-name "PongNoFrameskip-v4"  --use-pnn --n-columns 2 --pnn-paths "path_to_trained_model_from_previous_runs"  --use-gae   --num-processes 8 --num-steps 128 --num-mini-batch 4  --use-linear-lr-decay 
```

Works with minigrid environments. Pass 'MiniGrid-xyz' (change this to environment's name) as the argument for --env-name
