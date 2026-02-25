
# Wizard PPO 

This repository contains the practical implementation of a Bachelor Thesis at JKU for the Artificial Intelligence program.
It enables training and playing an agent for the card game Wizard using the Proximal Policy Optimization (PPO) algorithm developed by OpenAI.

The theoretical background and detailed explanations are provided in the thesis.

## Installation

It is recommended to use a conda environment to install the required dependencies.

```bash
conda create -n wizard_ppo python=3.11
conda activate wizard_ppo
pip install -r requirements.txt
```

## Start Training

To start training, run the following command in the root directory:

```python
python main.py 
```
All training parameters are defined in `parameter.yaml`.
Logs are saved automatically in the `log` directory.
TensorBoard will start automatically for monitoring the training progress.


## Pretrained model

In directory model two pretrained models are already provides. 

| Model | Training Iterations | Description                           |
|-------|---------------------|---------------------------------------|
| `model_20.pth` | 200 | model after 20% of training iteration |
| `model.pth` | 1000 | fully trained model                   |

You can use these models to test or continue training.

## Wizard GUI

To test the trained model a playable interface is provided.

This GUI is experimental! 
There may be display issues with cards and no guarantee of correctness is provided.

```python
python -m gui.main --model-path PATH/TO/MODEL
```