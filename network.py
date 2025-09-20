"""
All Networks used for Pytorch HIQL implementation

"""
from collections.abc import Callable
from functools import partial
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

def apply_variance_scaling_init(net, scale=1.0):
    fan = (net.weight.size(-2) + net.weight.size(-1)) / 2
    init_w = math.sqrt(scale / fan)
    net.weight.data.uniform_(-init_w, init_w)
    net.bias.data.fill_(0)

def custom_init_mlp(
    sizes: list[int],
    activation: nn.Module=nn.GELU,
    output_activation: nn.Module = nn.Identity,
    output_init_scaling: float = 1.0,
    dropout: float = None,
    layer_norm: bool = False,
):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            fc = nn.Linear(sizes[j], sizes[j + 1])
            apply_variance_scaling_init(fc, scale=output_init_scaling)
            if layer_norm and j > 0:
                layers += [nn.LayerNorm(sizes[j], eps=1e-6), fc, act()]
            else:
                layers += [fc, act()]
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        else:
            fc = nn.Linear(sizes[j], sizes[j + 1])
            apply_variance_scaling_init(fc, scale=output_init_scaling)
            if layer_norm:
                layers += [nn.LayerNorm(sizes[j], eps=1e-6), fc, act()]
            else:
                layers += [fc, act()]
    m = nn.Sequential(*layers)
    return m

class ResidualBlock(nn.Module):
    def __init__(self, inp_channel, num_features, max_pooling=True, activ='relu'):
        super().__init__()
        self.num_features = num_features
        self.max_pooling = max_pooling
        self.cnn1 = nn.Conv2d(inp_channel, num_features, 3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, padding=1) if max_pooling else nn.Identity()
        self.cnn2 = nn.Conv2d(num_features, num_features, 3, stride=1, padding=1)
        self.cnn3 = nn.Conv2d(num_features, num_features, 3, stride=1, padding=1)
        self.activ = getattr(F, activ)

    def forward(self, x):

        x1 = self.max_pool1(self.cnn1(x))
        block_inp = x1
        x2 = self.cnn2(self.activ(x1))
        x2 = self.cnn3(self.activ(x2))
        x2 += block_inp 
        return x2

class SimpleCNN(nn.Module):
    def __init__(self, state_dim, out_dim, activ='elu',concat_encoder=False):
        super().__init__()
        self.out_dim = out_dim
        self.zs_cnn1 = nn.Conv2d(state_dim, 32, 3, stride=2)
        self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
        self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
        self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
        input_dim = 2080 if concat_encoder else 800
        self.zs_lin = custom_init_mlp([input_dim, out_dim], output_activation=nn.ELU)
        self.activ = getattr(F, activ)

    def forward(self, state):
        state = state/255. - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
        return self.zs_lin(zs)

class ImpalaCNN(nn.Module):
    def __init__(self, state_dim, out_dim, activ='relu', concat_encoder=False):
        super().__init__()
        self.out_dim = out_dim
        self.zs_cnn1 = ResidualBlock(state_dim, 16, True)
        self.zs_cnn2 = ResidualBlock(16, 32, True)
        self.zs_cnn3 = ResidualBlock(32, 32, True)
        input_dim = 2*2048 if concat_encoder else 2048
        self.zs_lin = custom_init_mlp([input_dim, out_dim], output_activation=nn.GELU)
        self.activ = getattr(F, activ)
        
    def forward(self, state):
        # state = state/255. - 0.5
        state = state/255. # cube
        zs = self.zs_cnn1(state)
        zs = self.zs_cnn2(zs)
        zs = self.zs_cnn3(zs).reshape(state.shape[0], -1)
        return self.zs_lin(zs)

class LengthNormalize(nn.Module):
    """Length normalization layer.

    Normalizes the input along the last dimension to have a length of sqrt(dim).
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        dim = x.shape[-1]
        return x / norm * torch.sqrt(torch.tensor(dim, dtype=x.dtype, device=x.device))

class goal_rep(nn.Module):
    def __init__(self, state_dim=None, encoder_module=None, goal_dim=25, n_layer=2):
        super().__init__()
        self.gc_encoder = encoder_module(concat_encoder=True) if encoder_module else nn.Identity()
        input_dim = self.gc_encoder.out_dim if encoder_module else 2*state_dim
        layers = [512] * n_layer
        encoder = [custom_init_mlp(
            # sizes=[input_dim]+[512, 512, 512]+ [goal_dim], # one layer less  
            sizes=[input_dim]+layers+ [goal_dim], # one layer less  
            activation=nn.GELU,
            layer_norm=True
        )]
        encoder.append(LengthNormalize())
        self.encoder = nn.Sequential(*encoder)
        self.out_dim = goal_dim

    def forward(self, state, goal=None):
        # if self.gc_encoder is not None:
        #     state = self.gc_encoder(state)
        #     goal  = self.gc_encoder(goal)
        if goal is not None:
            input = torch.cat([state, goal], -1)
            input = self.gc_encoder(input)
            return self.encoder(input)
        else:
            input = self.gc_encoder(state)
            return self.encoder(input)

class GCEncoder(nn.Module):
    def __init__(self, state_encoder=None, goal_encoder=None, concat_encoder=None):
        super().__init__()

        self.state_encoder=state_encoder() if state_encoder is not None else None
        self.goal_encoder=goal_encoder() if goal_encoder is not None else None
        self.concat_encoder=concat_encoder() if concat_encoder is not None else None

    def forward(self, state, goal=None, goal_encoded=False):
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(state))
        if goal is not None:
            if goal_encoded:
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goal)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goal))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(torch.cat([state, goal], axis=-1)))
        reps = torch.concat(reps, axis=-1)
        return reps

class GCValue(nn.Module):
    def __init__(self, state_dim, gc_encoder=None):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, state_dim, gc_encoder):
                
                super().__init__()
                self.gc_encoder = gc_encoder
                input_dim = self.dummy_inp(state_dim) if gc_encoder is not None else state_dim[0]
                self.mlp_module = custom_init_mlp(
                                    # sizes=[input_dim]+[512, 512, 512]+ [1], # one layer less  
                                    sizes=[input_dim]+[512,512,512]+ [1], # one layer less  
                                    activation=nn.GELU,
                                    layer_norm=True)
                
            def dummy_inp(self, state_dim):
                with torch.no_grad():
                    if isinstance(state_dim, int):
                        state_dim = (state_dim,)
                    x = torch.rand([1, *state_dim])
                    out = self.gc_encoder(x, x)
                    return out.shape[-1]

            def forward(self, state, goal):
                x = self.gc_encoder(state, goal)
                x = self.mlp_module(x)
                return x
        self.q1 = ValueNetwork(state_dim, gc_encoder())
        self.q2 = ValueNetwork(state_dim, gc_encoder())
    
    def forward(self, state, goal):
        return self.q1(state, goal), self.q2(state, goal)

class Value(nn.Module):
    def __init__(self, state_dim, gc_encoder=None, n_layer=2):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, state_dim, gc_encoder, n_layer=2):
                
                super().__init__()
                self.gc_encoder = gc_encoder if gc_encoder is not None else nn.Identity()
                input_dim = self.dummy_inp(state_dim) if gc_encoder is not None else state_dim
                layers = n_layer*[512]
                self.mlp_module = custom_init_mlp(
                                    # sizes=[input_dim]+[512, 512, 512]+ [1], # one layer less  
                                    sizes=[input_dim]+layers+ [1], # one layer less  
                                    activation=nn.GELU,
                                    layer_norm=True)
                
            def dummy_inp(self, state_dim):
                with torch.no_grad():
                    x = torch.rand([1, *state_dim])
                    out = self.gc_encoder(x, x)
                    return out.shape[-1]

            def forward(self, state, goal):
                inputs = [state]
                if goal is not None:
                    inputs.append(goal)
                inputs = torch.concat(inputs, -1)
                x = self.gc_encoder(inputs)
                x = self.mlp_module(x)
                return x
        self.q1 = ValueNetwork(state_dim, None, n_layer=n_layer)
        self.q2 = ValueNetwork(state_dim, None, n_layer=n_layer)
    
    def forward(self, state, goal):
        return self.q1(state, goal), self.q2(state, goal)

class GCActor(nn.Module):
    def __init__(
        self,
        state_dim:int,
        action_dim: int,
        gc_encoder:nn.Module,
        state_dependent_std: bool = False,
        const_std: bool = True,
        log_std_min: float = -5.0,      # Clip log std lower bound
        log_std_max: float = 2.0,       # Clip log std upper bound
        n_layer=2
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gc_encoder = gc_encoder() if gc_encoder is not None else None
        input_dim = self.dummy_inp(state_dim) if gc_encoder is not None else state_dim
        layers = n_layer * [512]
        self.actor_net = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[input_dim] + layers, # 2 layers 
            activation=nn.GELU,
            output_activation=nn.GELU,
        )
        self.mean_net = nn.Linear(512, action_dim)
        apply_variance_scaling_init(self.mean_net, scale=0.01)
        if self.state_dependent_std:
            self.log_std_net = nn.Linear(512, action_dim)
            apply_variance_scaling_init(self.log_std_net, scale=0.01)
        else:
            if not const_std:
                self.log_stds = nn.Parameter(torch.zeros(action_dim))
    
    def dummy_inp(self, state_dim):
        with torch.no_grad():
            if isinstance(state_dim, int):
                state_dim = (state_dim,)
            x = torch.rand([1, *state_dim])
            out = self.gc_encoder(x, x)
            return out.shape[-1]

    def forward(
            self, 
            state:torch.Tensor, 
            goal:torch.Tensor,
            goal_encoded=False,    
            temperature=1.0):

        if self.gc_encoder is not None:
            inputs = self.gc_encoder(state, goal, goal_encoded=goal_encoded)
        else:
            inputs = [state]
            if goal is not None:
                inputs.append(goal)
            inputs = torch.concat(inputs, -1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = torch.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)

        dist = D.Independent(D.Normal(means, (torch.exp(log_stds)*temperature).clamp(min=0.0001)), 
                             reinterpreted_batch_ndims=1)
        # Tanh Squah is not used 
        return dist

    def act(
            self, 
            state:torch.Tensor, 
            goal:torch.Tensor,
            goal_encoded:bool=False,
            temperature=1.0):
        
        # Temerature is 0 while training 
        dist = self.forward(state, goal, goal_encoded, temperature)
        return dist



class WorldModel_MrHi(nn.Module):
    def __init__(self, 
                 zs_dim:int,
                 action_dim:int, 
                 state_encoder:nn.Module,
                 goal_rep_fn:nn.Module,
                 za_dim:int = 256,
                 sigmoid:bool=False,
                 ln:bool=False,
                 num_bins=21):
        """
        This is the version used for two Hot entropy loss
        """
        super().__init__()
        self.zs = state_encoder()
        self.goal_rep = goal_rep_fn()
        inp_dim = self.zs.out_dim
        goal_dim = self.goal_rep.out_dim
        self.sigmoid = sigmoid
        self.ln = LengthNormalize() if ln else nn.Identity()
        self.num_bins = num_bins

        self.model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[inp_dim] + [zs_dim],
            activation=nn.GELU,
            layer_norm=False
        )
        self.rew_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim + goal_dim] + [num_bins],
            activation=nn.GELU,
            layer_norm=False
        )
        
        self.za = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[action_dim] + [za_dim],
            activation=nn.GELU,
            output_activation=nn.GELU,
            layer_norm=False
        )
        
        self.zsa = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+za_dim] + [512, 512] + [zs_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.low_inv_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[2*zs_dim] + [512] + [action_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.high_inv_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+goal_dim] + [512] + [action_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )
    
    def forward(self, zs, action):
        za = self.za(action)
        x = torch.cat([zs, za], -1)
        return self.zsa(x)

    def phi(self, zs, goal_zs):
        return self.goal_rep(torch.cat([zs, goal_zs], -1))

    def model_all(self, zs, next_zs, goal_zs, action):
        # dynamics model 
        zsa = self.forward(zs, action) # Latent
        goal_rep = self.phi(zs, goal_zs)
        pred_next_zs = self.model(zsa)
        
        # Low level Inverse Dynamics Model
        pred_action_low = self.low_inv_model(torch.cat([zs, next_zs], -1))
        
        # Predict reward / return
        pred_reward = self.rew_model(torch.cat([zsa, goal_rep], -1))  
        pred_action_high = self.high_inv_model(torch.cat([zs, goal_rep], -1))

        return pred_next_zs, pred_action_low, pred_action_high, pred_reward



class WorldModel_MrHiAc(nn.Module):
    def __init__(self, 
                 zs_dim:int,
                 action_dim:int, 
                 state_encoder:nn.Module,
                 goal_rep_fn:nn.Module,
                 za_dim:int = 256,
                 sigmoid:bool=False,
                 ln:bool=False,
                 num_bins=21,
                 act_chunk=5):
        """
        This is the version used for two Hot entropy loss
        """
        super().__init__()
        self.zs = state_encoder()
        self.goal_rep = goal_rep_fn()
        inp_dim = self.zs.out_dim
        goal_dim = self.goal_rep.out_dim
        self.sigmoid = sigmoid
        self.ln = LengthNormalize() if ln else nn.Identity()
        self.num_bins = num_bins
        self.act_chunk = act_chunk if act_chunk > 0 else 1

        self.model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[inp_dim] + [zs_dim],
            activation=nn.GELU,
            layer_norm=False
        )
        self.rew_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim + goal_dim] + [num_bins],
            activation=nn.GELU,
            layer_norm=False
        )
        
        self.za = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[action_dim] + [za_dim],
            activation=nn.GELU,
            output_activation=nn.GELU,
            layer_norm=False
        )
        
        self.zsa = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+za_dim] + [512, 512] + [zs_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.low_inv_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[2*zs_dim] + [512] + [action_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.high_inv_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+goal_dim] + [512] + [self.act_chunk * action_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )
    
    def forward(self, zs, action):
        za = self.za(action)
        x = torch.cat([zs, za], -1)
        return self.zsa(x)

    def phi(self, zs, goal_zs):
        return self.goal_rep(torch.cat([zs, goal_zs], -1))

    # def model_all(self, zs, next_zs, goal_zs, action):
    #     # dynamics model 
    #     zsa = self.forward(zs, action) # Latent
    #     goal_rep = self.phi(zs, goal_zs)
    #     pred_next_zs = self.model(zsa)
        
    #     # Low level Inverse Dynamics Model
    #     pred_action_low = self.low_inv_model(torch.cat([zs, next_zs], -1))
        
    #     # Predict reward / return
    #     pred_reward = self.rew_model(torch.cat([zsa, goal_rep], -1))  
    #     pred_action_high = self.high_inv_model(torch.cat([zs, goal_rep], -1))

    #     return pred_next_zs, pred_action_low, pred_action_high, pred_reward
    
    def model_all(self, zs, next_zs, goal_zs, action, stop_hl_grads : bool = False):
        # LL Dyn model 
        zsa = self.forward(zs, action) # Latent
        pred_next_zs = self.model(zsa)
        
        # LL IDM
        pred_action_low = self.low_inv_model(torch.cat([zs, next_zs], -1))
        

        if stop_hl_grads:
          # <-- stopping gradients for everything after here -->
          # see the all the detach() in the lines below
          goal_rep = self.phi(zs.detach(), goal_zs.detach())
          # GC-Rew
          pred_reward = self.rew_model(torch.cat([zsa.detach(), goal_rep], -1))  
          # HL IDM
          pred_action_high = self.high_inv_model(torch.cat([zs.detach(), goal_rep], -1))
          
        else:
          # Predict reward / return
          goal_rep = self.phi(zs, goal_zs)
          pred_reward = self.rew_model(torch.cat([zsa, goal_rep], -1))  
          pred_action_high = self.high_inv_model(torch.cat([zs, goal_rep], -1))
          
        return pred_next_zs, pred_action_low, pred_action_high, pred_reward #, pred_future_zs


class WorldModel_MrHiAcHd(nn.Module):
    def __init__(self, 
                 zs_dim:int,
                 action_dim:int, 
                 state_encoder:nn.Module,
                 goal_rep_fn:nn.Module,
                 za_dim:int = 256,
                 sigmoid:bool=False,
                 ln:bool=False,
                 num_bins=21,
                 act_chunk=5):
        """
        This is the version used for two Hot entropy loss
        """
        super().__init__()
        self.zs = state_encoder()
        self.goal_rep = goal_rep_fn()
        inp_dim = self.zs.out_dim
        goal_dim = self.goal_rep.out_dim
        self.sigmoid = sigmoid
        self.ln = LengthNormalize() if ln else nn.Identity()
        self.num_bins = num_bins
        self.act_chunk = act_chunk if act_chunk > 0 else 1

        self.model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[inp_dim] + [zs_dim],
            activation=nn.GELU,
            layer_norm=False
        )
        self.rew_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim + goal_dim] + [num_bins],
            activation=nn.GELU,
            layer_norm=False
        )
        
        self.za = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[action_dim] + [za_dim],
            activation=nn.GELU,
            output_activation=nn.GELU,
            layer_norm=False
        )
        
        self.zsa = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+za_dim] + [512, 512] + [zs_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.low_inv_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[2*zs_dim] + [512] + [action_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.high_inv_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+goal_dim] + [512] + [self.act_chunk * action_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )

        self.high_dyn_model = custom_init_mlp(
            # sizes=[input_dim] + [512,512,512],
            sizes=[zs_dim+goal_dim] + [512] + [zs_dim],
            activation=nn.GELU,
            layer_norm=True
            # output_activation=nn.GELU,
        )
    
    def forward(self, zs, action):
        za = self.za(action)
        x = torch.cat([zs, za], -1)
        return self.zsa(x)

    def phi(self, zs, goal_zs):
        return self.goal_rep(torch.cat([zs, goal_zs], -1))

    # def model_all(self, zs, next_zs, goal_zs, action, action_chunk):
    #     # dynamics model 
    #     zsa = self.forward(zs, action) # Latent
    #     goal_rep = self.phi(zs, goal_zs)
    #     pred_next_zs = self.model(zsa)
        
    #     # Low level Inverse Dynamics Model
    #     pred_action_low = self.low_inv_model(torch.cat([zs, next_zs], -1))
        
    #     # Predict reward / return
    #     pred_reward = self.rew_model(torch.cat([zsa, goal_rep], -1))  
    #     pred_action_high = self.high_inv_model(torch.cat([zs, goal_rep], -1))

    #     pred_phi_next = self.high_dyn_model(torch.cat([goal_rep, action_chunk], -1))  # Phi_target(zs_t+n, zg)
    #     return pred_next_zs, pred_action_low, pred_action_high, pred_reward, pred_phi_next

    def model_all(self, zs, next_zs, goal_zs, action, stop_hl_grads : bool = False):
        # LL Dyn model 
        zsa = self.forward(zs, action) # Latent
        pred_next_zs = self.model(zsa)
        
        # LL IDM
        pred_action_low = self.low_inv_model(torch.cat([zs, next_zs], -1))
        

        if stop_hl_grads:
          # <-- stopping gradients for everything after here -->
          # see the all the detach() in the lines below
          goal_rep = self.phi(zs.detach(), goal_zs.detach())
          
          # GC-Rew
          pred_reward = self.rew_model(torch.cat([zsa.detach(), goal_rep], -1))  
          # HL IDM
          pred_action_high = self.high_inv_model(torch.cat([zs.detach(), goal_rep], -1))
          # Missing: HL Dyn
          pred_future_zs = self.high_dyn_model(torch.cat([zs.detach(), goal_rep], -1))
        else:
          # Predict reward / return
          goal_rep = self.phi(zs, goal_zs)
          pred_reward = self.rew_model(torch.cat([zsa, goal_rep], -1))  
          pred_action_high = self.high_inv_model(torch.cat([zs, goal_rep], -1))
          # Missing: HL Dyn
          pred_future_zs = self.high_dyn_model(torch.cat([zs, goal_rep], -1))

        return pred_next_zs, pred_action_low, pred_action_high, pred_reward, pred_future_zs


"""
comments about pred_action_high and pred_phi_subgoal share the same input Next encoder V2 of MrHiAcSd
"""
# Will be moved to Utils
def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

"""
if __name__ == "__main__":
    state_dim = (3, 64, 64)
    action_dim = 5
    state = torch.rand([10,*state_dim])
    goal = torch.rand([10,*state_dim])
    action = torch.rand([10,action_dim])
    
    enc = 'simple'
    if enc == 'impala':
        encoder = ImpalaCNN
    else:
        encoder = SimpleCNN
    
    # goalnet = goal_rep(state_dim[0], 
    #                    encoder_module=encoder(state_dim=state_dim[0], 
    #                                           out_dim=512, 
    #                                           concat_encoder=True)
    #                     )
    
    # print(num_parameters(goalnet))
    # # Impala 2.9 or Simple 1.9M
    # out = goalnet(state, goal) 
    # print(out.shape)
    ENCODERS = {
            "impala": ImpalaCNN,
            "simple": SimpleCNN,
            "none"  : None
        }

        # pick the encoder class
    encoder_cls = ENCODERS.get(enc, None)
    Encoder_module = partial(encoder_cls, state_dim=state_dim[0], out_dim=512)
    zs_dim = 512
    goal_rep_fn = partial(goal_rep, state_dim=512, 
                        encoder_module=None)
    
    wm = WorldModel_MrHi(zs_dim=512, action_dim=action_dim, 
                    state_encoder=Encoder_module,
                    goal_rep_fn=goal_rep_fn, ln=True, num_bins=21)
    
    zs, a_los, a_high, r = wm.model_all(wm.zs(state), wm.zs(state), wm.zs(goal), action)

    print(num_parameters(wm))
    check = 1

    
    goal_rep_fn = partial(goal_rep, state_dim=state_dim[0], 
                        encoder_module=Encoder_module)
    # 2.2M parameter
    
    # 2.1M parameter
    gc_encoder = partial(GCEncoder, state_encoder = Encoder_module,
                            concat_encoder=goal_rep_fn)
    
    # 0.87
    # gc_encoder = GCEncoder(state_encoder=SimpleCNN(state_dim[0], 512), 
    #                         goal_encoder=SimpleCNN(state_dim[0], 512)) 
    # 1.1M parameter
    # gc_encoder = GCEncoder(concat_encoder=SimpleCNN(state_dim[0], 512, concat_encoder=True))

    # print(num_parameters(gc_encoder))
    # out = gc_encoder(state, goal)
    value = GCValue(state_dim, gc_encoder)
    print(num_parameters(value))
    out = value(state, goal)
    # print(out)

    # Actor 
    actor = GCActor(state_dim, 
                    action_dim=25, 
                    gc_encoder=gc_encoder,
                    state_dependent_std=False,
                    const_std=True)
    
    print(num_parameters(actor))
    out = actor(state, goal)
    print(out.mean.shape)
    """