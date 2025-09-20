import dataclasses
import pprint
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a nested dictionary with dot notation (e.g., 'policy.lr')."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enforce_dataclass_type(dataclass: dataclasses.dataclass):
    for field in dataclasses.fields(dataclass):
        setattr(dataclass, field.name, field.type(getattr(dataclass, field.name)))


def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp):
        c.__dict__[field.name] = getattr(hp, field.name)

def log_util(metrics, t):
    # Group metrics by their main key (encoder/critic/actor)
    groups = {}
    for k, v in metrics.items():
        metric_type = k.split('/')[0]
        group = k.split('/')[-1].split('_')[0]
        metric = '_'.join(k.split('/')[-1].split('_')[1:])  # handles e.g. actor_Bcloss
        if group not in groups:
            groups[group] = []
        groups[group].append((metric, v))
    n = 35
    # Print each group in a formatted line
    log_text = ''.join(['-']*n + [' ' + metric_type.capitalize()+f" Metric: {t} " ] + ['-']*n + ['\n'])
    for group in ['encoder', 'critic', 'actor', 'Value', 'Hactor', 'Lactor']:
        if group in groups:
            metrics_str = '\t| '.join(f"{m}={val:.4f}" for m, val in groups[group])
            log_text += f"{group.capitalize()}\t: {metrics_str}\n"
    log_text += ''.join(['-']*2*(n+10) + ['\n'])
    
    return log_text

class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file


    def log_print(self, x):
        with open(self.log_file, 'a') as f:
            if isinstance(x, str):
                print(x)
                f.write(x+'\n')
            else:
                pprint.pprint(x)
                pprint.pprint(x, f)


    def title(self, text: str):
        self.log_print('-'*40)
        self.log_print(text)
        self.log_print('-'*40)


# Takes the formatted results and returns a dictionary of env -> (timesteps, seed).
def results_to_numpy(file: str='../results/gym_results.txt'):
    results = {}

    for line in open(file):
        if '----' in line:
            continue
        if 'Timestep' in line:
            continue
        if 'Env:' in line:
            env = line.split(' ')[1][:-1]
            results[env] = []
        else:
            timestep = []
            for seed in line.split('\t')[1:]:
                if seed != '':
                    seed = seed.replace('\n', '')
                    timestep.append(float(seed))
            results[env].append(timestep)

    for k in results:
        results[k] = np.array(results[k])
        print(k, results[k].shape)

    return results



class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

