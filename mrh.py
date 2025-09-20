import copy
import dataclasses
from typing import Dict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import buffer
import bufferv3 as bufferv2
from network import *
import utils
from largebuffer import ReplayBufferBatch
import random

@dataclasses.dataclass
class Hyperparameters:
    batched_buffer:bool = False
    chunks:int = 3
    batch_size: int = 256
    buffer_size: int = 1e6
    gc_negative:bool = True
    discount: float = 0.99
    target_update_freq: int = 250
    storage_cpu:bool = False
    tau:float = 0.005
    subgoal_step:int=25
    comp_return:bool = True
    # encoder 
    encoder_module:str = 'impala' # Impala, Simple or None
    enc_version:str = 'V2'  # V2, V2R
    enc_horizon:int = 5
    Q_horizon:int = 1
    expectile:float = 0.7
    sigmoid_fn:bool = False 
    lengthnorm_fn:bool = False

    goal_dim:int = 10
    pixel_augs:bool = False
    
    dyn_weight:float = 1.0
    zs_dim:int = 512
    encoder_lr:float = 1e-4
    encoder_wd:float = 1e-4

    value_lr:float = 3e-4
    value_wd:float = 1e-4

    high_actor_lr:float = 3e-4
    high_actor_wd:float = 1e-4
    high_alpha:float = 3.0

    low_actor_lr:float = 3e-4
    low_actor_wd:float = 1e-4
    low_alpha:float = 3.0
    
    state_dependent_std:bool = False
    const_std:bool = True

    bufferV2:bool=False
    vg_cur:float = 0.2
    vg_rand:float = 0.3 
    vg_traj:float = 0.5

    ag_cur:float = 0.0
    ag_rand:float = 0.0 
    ag_traj:float = 1.0

    two_hot_loss:bool= False
    num_bins:int = 21

    act_chunk:int = 5

    # 
    n_layer:int = 2
    total_train_steps:int=500000
    cosine_scheduler:bool = False
    pause_encoder:bool = False

    def __post_init__(self): utils.enforce_dataclass_type(self)

class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool,
        device: torch.device, history: int=1, hp: Dict={}):
        self.name = "hiql"

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        self.act_chunk = self.act_chunk if self.enc_version in ['MrHiAc','MrHiAcHd', 'MrHiAcSg'] else 1

        self.hd = True if self.enc_version in ['MrHiAcHd'] else False
        self.sg = True if self.enc_version in ['MrHiAcSg'] else False

        if self.hp.batched_buffer:
            print("Using a Chunk Buffer")
            self.replay_buffer = ReplayBufferBatch(self.batch_size, weight=None, gamma=self.discount, 
                                          storage_cpu=self.storage_cpu, stack=history, chunks=self.hp.chunks,
                                          sub_goal=self.subgoal_step)

        else:
            comp_return = self.comp_return
            
            if self.bufferV2:
                self.replay_buffer = bufferv2.ReplayBuffer(self.batch_size, weight=None, gamma=self.discount, 
                                                       value_current_p=self.vg_cur, actor_current_p=self.ag_cur,
                                                       value_random_p=self.vg_rand, actor_random_p=self.ag_rand,
                                                       value_traj_p=self.vg_traj, actor_traj_p=self.ag_traj,
                                                       storage_cpu=self.storage_cpu, stack=history, 
                                                       sub_goal=self.subgoal_step, comp_return=comp_return, 
                                                       act_chunk=self.act_chunk, hd=self.hd, sg=self.sg)
        
            else:
                print("Using a Normal Buffer")
                self.replay_buffer = buffer.ReplayBuffer(self.batch_size, weight=None, gamma=self.discount, 
                                                        storage_cpu=self.storage_cpu, stack=history, sub_goal=self.subgoal_step,
                                                        comp_return=comp_return, act_chunk=self.act_chunk)
        
        ENCODERS = {
            "impala": ImpalaCNN,
            "simple": SimpleCNN,
            "none"  : None
        }

        # pick the encoder class
        encoder_cls = ENCODERS.get(self.encoder_module, None)
        
        #### Model ####
        Encoder_module = partial(encoder_cls, state_dim=obs_shape[0], out_dim=self.zs_dim)
        
        goal_rep_fn = partial(goal_rep, state_dim=self.zs_dim, 
                            encoder_module=None, goal_dim=self.goal_dim, n_layer=self.n_layer)
        # choice of world model
        if self.enc_version == 'MrHi':
            print("Using MrHi")
            num_bins = self.num_bins if self.two_hot_loss else 1
            self.encoder = WorldModel_MrHi(zs_dim=self.zs_dim, action_dim=action_dim, 
                                        state_encoder=Encoder_module, 
                                        goal_rep_fn=goal_rep_fn, sigmoid=self.sigmoid_fn, 
                                        ln=self.lengthnorm_fn, num_bins=num_bins).to(self.device)
        elif self.enc_version == 'MrHiAc':
            print("Using MrHiAC")
            num_bins = self.num_bins if self.two_hot_loss else 1
            self.encoder = WorldModel_MrHiAc(zs_dim=self.zs_dim, action_dim=action_dim, 
                                        state_encoder=Encoder_module, 
                                        goal_rep_fn=goal_rep_fn, sigmoid=self.sigmoid_fn, 
                                        ln=self.lengthnorm_fn, num_bins=num_bins, 
                                        act_chunk=self.act_chunk).to(self.device)
        elif self.enc_version == 'MrHiAcHd':
            print("Using MrHiAC-High_dyna")
            num_bins = self.num_bins if self.two_hot_loss else 1
            self.encoder = WorldModel_MrHiAcHd(zs_dim=self.zs_dim, action_dim=action_dim, 
                                        state_encoder=Encoder_module, 
                                        goal_rep_fn=goal_rep_fn, sigmoid=self.sigmoid_fn, 
                                        ln=self.lengthnorm_fn, num_bins=num_bins, 
                                        act_chunk=self.act_chunk).to(self.device)
        else:
            raise NotImplementedError(f"Unknown world model: {self.enc_version}")

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        if self.cosine_scheduler:
            self.encoder_scheduler = CosineAnnealingLR(self.encoder_optimizer, T_max=self.total_train_steps)
        self.encoder_target = copy.deepcopy(self.encoder)
        
        # In this setting we are going to use goal encoder
        Encoder_module = None
        goal_rep_fn = nn.Identity
        value_encoder_def = None
        low_actor_encoder_def = None
        high_actor_encoder_def = None
        
        # Value
        self.value = Value(state_dim=self.zs_dim+self.goal_dim, gc_encoder=value_encoder_def, n_layer=self.n_layer).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.value_target = copy.deepcopy(self.value)
        
        # Low-level policy
        self.low_actor = GCActor(state_dim=self.zs_dim+self.goal_dim, action_dim=action_dim, gc_encoder=low_actor_encoder_def, 
                                 state_dependent_std=False, const_std=True, n_layer=self.n_layer).to(self.device)
        self.low_actor_optimizer = torch.optim.Adam(self.low_actor.parameters(), lr=self.low_actor_lr)

        # High Level Policy
        self.high_actor = GCActor(state_dim=2*self.zs_dim, action_dim=self.goal_dim, gc_encoder=high_actor_encoder_def, 
                                 state_dependent_std=False, const_std=True, n_layer=self.n_layer).to(self.device)
        self.high_actor_optimizer = torch.optim.Adam(self.high_actor.parameters(), lr=self.high_actor_lr)
        
        # Environment properties
        self.pixel_obs = pixel_obs
        self.state_shape = obs_shape # This includes history, horizon, channels, etc.
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action
        if self.two_hot_loss:
            self.two_hot = TwoHot(self.device, -1, 1, self.num_bins)

        # Tracked values
        self.reward_scale, self.target_reward_scale = 1, 1
        self.training_steps = 0
        # ADDED 
        self.encoder_metric = { 'train/encoder_loss' : 0}
        self.history = history
    
    def train_encoder(self, batch):
        """
        Training the Encoder
        """
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(
                batch["next_state"].reshape(-1, *self.state_shape)
            ).reshape(batch["state"].shape[0], -1, self.zs_dim)
        
        goal_zs = self.encoder.zs(batch['goal'])
        pred_zs = self.encoder.zs(batch["state"][:, 0])

        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.
        dyn_losses, inv_losses, return_losses = [], [], []

        for i in range(self.enc_horizon):
            pred_zs, pred_a, pred_r = self.encoder.model_all(pred_zs, encoder_target[:,i], 
                                                             goal_zs, batch['action'][:,i])
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            inv_loss = masked_mse(pred_a, batch['action'][:,i], prev_not_done)
            
            if not self.two_hot_loss:
                return_loss = masked_mse(pred_r, batch['reward'][:,i].reshape(-1,1), prev_not_done) 
            else:
                # (F.softmax(pred_r, dim=-1) * self.two_hot.bins).sum(1, keepdim=True) - batch['reward'][:, i].unsqueeze(-1)
                return_loss = (self.two_hot.cross_entropy_loss(pred_r, batch['reward'][:,i].reshape(-1, 1)) * prev_not_done).mean()
            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + 0.1 * return_loss + 0.1 * inv_loss
            prev_not_done = batch['not_done'][:,i].reshape(-1,1) * prev_not_done
            dyn_losses.append(dyn_loss.item())
            inv_losses.append(inv_loss.item())
            return_losses.append(return_loss.item())

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()

        if self.cosine_scheduler:
            self.encoder_scheduler.step()
        
        metrics = {
                    'encoder_loss': encoder_loss.item(),
                    # 'dyn_loss_last': dyn_losses[-1],
                    # 'inv_loss_last': inv_losses[-1],
                    # 'return_loss_last': return_losses[-1],
                    'encoder_DynLoss': np.mean(dyn_losses),
                    'encoder_InvLoss': np.mean(inv_losses),
                    'encoder_VaLoss': np.mean(return_losses),
                }
        return metrics
    
    def train_encoder_MrHi(self, batch):
        """
        Training the Encoder for MrHi version
        """
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(
                batch["next_state"].reshape(-1, *self.state_shape)
            ).reshape(batch["state"].shape[0], -1, self.zs_dim)
        
        goal_zs = self.encoder.zs(batch['goal'])
        pred_zs = self.encoder.zs(batch["state"][:, 0])

        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.
        dyn_losses, low_inv_losses, high_inv_losses, return_losses = [], [], [], []

        for i in range(self.enc_horizon):
            pred_zs, pred_al, pred_ah, pred_r = self.encoder.model_all(pred_zs, encoder_target[:,i], 
                                                             goal_zs, batch['action'][:,i])
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            inv_loss_l = masked_mse(pred_al, batch['action'][:,i], prev_not_done)
            inv_loss_h = masked_mse(pred_ah, batch['action'][:,i], prev_not_done)
            
            if not self.two_hot_loss:
                return_loss = masked_mse(pred_r, batch['reward'][:,i].reshape(-1,1), prev_not_done) 
            else:
                # (F.softmax(pred_r, dim=-1) * self.two_hot.bins).sum(1, keepdim=True) - batch['reward'][:, i].unsqueeze(-1)
                return_loss = (self.two_hot.cross_entropy_loss(pred_r, batch['reward'][:,i].reshape(-1, 1)) * prev_not_done).mean()
            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + 0.1 * return_loss + 0.1 * inv_loss_l + 0.1 * inv_loss_h
            prev_not_done = batch['not_done'][:,i].reshape(-1,1) * prev_not_done
            dyn_losses.append(dyn_loss.item())
            low_inv_losses.append(inv_loss_l.item())
            high_inv_losses.append(inv_loss_h.item())
            return_losses.append(return_loss.item())

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()

        if self.cosine_scheduler:
            self.encoder_scheduler.step()
        
        metrics = {
                    'encoder_loss': encoder_loss.item(),
                    # 'dyn_loss_last': dyn_losses[-1],
                    # 'inv_loss_last': inv_losses[-1],
                    # 'return_loss_last': return_losses[-1],
                    'encoder_DynLoss': np.mean(dyn_losses),
                    'encoder_LInvLoss': np.mean(low_inv_losses),
                    'encoder_HInvLoss': np.mean(high_inv_losses),
                    'encoder_VaLoss': np.mean(return_losses),
                }
        return metrics
    
    def train_encoder_MrHiAc(self, batch):
        """
        Training the Encoder for MrHi version
        """
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(
                batch["next_state"].reshape(-1, *self.state_shape)
            ).reshape(batch["state"].shape[0], -1, self.zs_dim)
        
        goal_zs = self.encoder.zs(batch['goal'])
        pred_zs = self.encoder.zs(batch["state"][:, 0])

        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.
        dyn_losses, low_inv_losses, high_inv_losses, return_losses = [], [], [], []

        for i in range(self.enc_horizon):
            pred_zs, pred_al, pred_ah, pred_r = self.encoder.model_all(pred_zs, encoder_target[:,i], 
                                                             goal_zs, batch['action'][:,i])
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            inv_loss_l = masked_mse(pred_al, batch['action'][:,i], prev_not_done)
            inv_loss_h = masked_mse(pred_ah, batch['action'][:,i:i+self.act_chunk].reshape(-1, self.action_dim * self.act_chunk), prev_not_done)
            
            if not self.two_hot_loss:
                return_loss = masked_mse(pred_r, batch['reward'][:,i].reshape(-1,1), prev_not_done) 
            else:
                # (F.softmax(pred_r, dim=-1) * self.two_hot.bins).sum(1, keepdim=True) - batch['reward'][:, i].unsqueeze(-1)
                return_loss = (self.two_hot.cross_entropy_loss(pred_r, batch['reward'][:,i].reshape(-1, 1)) * prev_not_done).mean()
            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + 0.1 * return_loss + 0.1 * inv_loss_l + 0.1 * inv_loss_h
            prev_not_done = batch['not_done'][:,i].reshape(-1,1) * prev_not_done
            dyn_losses.append(dyn_loss.item())
            low_inv_losses.append(inv_loss_l.item())
            high_inv_losses.append(inv_loss_h.item())
            return_losses.append(return_loss.item())

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()

        if self.cosine_scheduler:
            self.encoder_scheduler.step()

        metrics = {
                    'encoder_loss': encoder_loss.item(),
                    # 'dyn_loss_last': dyn_losses[-1],
                    # 'inv_loss_last': inv_losses[-1],
                    # 'return_loss_last': return_losses[-1],
                    'encoder_DynLoss': np.mean(dyn_losses),
                    'encoder_LInvLoss': np.mean(low_inv_losses),
                    'encoder_HInvLoss': np.mean(high_inv_losses),
                    'encoder_VaLoss': np.mean(return_losses),
                }
        return metrics

    def train_encoder_MrHiAcHd(self, batch):
        """
        Training the Encoder for MrHi version
        """
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(
                batch["next_state"].reshape(-1, *self.state_shape)
            ).reshape(batch["state"].shape[0], -1, self.zs_dim)
            zs_tn = self.encoder_target.zs(
                batch["next_state_extra"].reshape(-1, *self.state_shape)
            ).reshape(batch["state"].shape[0], -1, self.zs_dim)
            zg_t = self.encoder_target.zs(batch['goal'])    
            
        
        goal_zs = self.encoder.zs(batch['goal'])
        pred_zs = self.encoder.zs(batch["state"][:, 0])

        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.
        dyn_losses, low_inv_losses, high_inv_losses, return_losses, high_dyn_losses = [], [], [], [], []

        for i in range(self.enc_horizon):
            # Detaching the phi_zs_{t+n}
            phi_tn = self.encoder_target.phi(zs_tn[:, i], zg_t).detach()
            act_chunk = batch['action'][:,i:i+self.act_chunk].reshape(-1, self.action_dim * self.act_chunk).detach()
            pred_zs, pred_al, pred_ah, pred_r, pred_phi = self.encoder.model_all(pred_zs, encoder_target[:,i], 
                                                             goal_zs, batch['action'][:,i], act_chunk)
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            inv_loss_l = masked_mse(pred_al, batch['action'][:,i], prev_not_done)
            inv_loss_h = masked_mse(pred_ah, act_chunk, prev_not_done)
            
            if not self.two_hot_loss:
                return_loss = masked_mse(pred_r, batch['reward'][:,i].reshape(-1,1), prev_not_done) 
            else:
                # (F.softmax(pred_r, dim=-1) * self.two_hot.bins).sum(1, keepdim=True) - batch['reward'][:, i].unsqueeze(-1)
                return_loss = (self.two_hot.cross_entropy_loss(pred_r, batch['reward'][:,i].reshape(-1, 1)) * prev_not_done).mean()
            
            high_dyn_loss = masked_mse(pred_phi, phi_tn, prev_not_done).mean()

            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + 0.1 * return_loss + 0.1 * inv_loss_l + 0.1 * inv_loss_h + 0.1* high_dyn_loss
            prev_not_done = batch['not_done'][:,i].reshape(-1,1) * prev_not_done
            dyn_losses.append(dyn_loss.item())
            low_inv_losses.append(inv_loss_l.item())
            high_inv_losses.append(inv_loss_h.item())
            return_losses.append(return_loss.item())
            high_dyn_losses.append(high_dyn_loss.item())

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()

        if self.cosine_scheduler:
            self.encoder_scheduler.step()

        metrics = {
                    'encoder_loss': encoder_loss.item(),
                    # 'dyn_loss_last': dyn_losses[-1],
                    # 'inv_loss_last': inv_losses[-1],
                    # 'return_loss_last': return_losses[-1],
                    'encoder_DynLoss': np.mean(dyn_losses),
                    'encoder_LInvLoss': np.mean(low_inv_losses),
                    'encoder_HInvLoss': np.mean(high_inv_losses),
                    'encoder_VaLoss': np.mean(return_losses),
                    'encoder_HDynLoss': np.mean(high_dyn_losses),
                }
        return metrics    


    def train_critic(self, batch):
        
        with torch.no_grad():
            next_zs = self.encoder_target.zs(batch['next_state'])
            zs = self.encoder_target.zs(batch['state'])
            value_goal_zs = self.encoder_target.zs(batch['value_goal'])
            phi_next_zs = self.encoder_target.phi(next_zs, value_goal_zs)
            phi_zs = self.encoder_target.phi(zs, value_goal_zs)
            
            next_v1_t, next_v2_t = self.value_target(next_zs, phi_next_zs)
            next_v_t = torch.minimum(next_v1_t, next_v2_t)
            q = batch['n-reward'] + batch['term_discount'] * next_v_t
            (v1_t, v2_t) = self.value_target(zs, phi_zs)
            v_t = (v1_t + v2_t) / 2
            adv = q - v_t

            q1 = batch['n-reward'] + batch['term_discount'] * next_v1_t
            q2 = batch['n-reward'] + batch['term_discount'] * next_v2_t

            zs = self.encoder.zs(batch['state'])
            value_goal_zs = self.encoder.zs(batch['value_goal'])
            phi_zs = self.encoder.phi(zs, value_goal_zs)
        
        v1, v2 = self.value(zs, phi_zs)
        v = (v1 + v2) / 2

        value_loss1 = expectile_loss(adv, q1 - v1, self.expectile).mean() # [Batch_size X 1] - [Batch_size X 1]
        value_loss2 = expectile_loss(adv, q2 - v2, self.expectile).mean()
        value_loss = value_loss1 + value_loss2

        # optimizer 
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()
        return {
            'train/Value_loss': value_loss.item(),
            'train/Value_mean': v.mean().item(),
            'train/Value_max': v.max().item(),
            'train/Value_min': v.min().item(),
        }

    def train_low_actor(self, batch):
        """Compute the low-level actor loss."""
        with torch.no_grad():
            zs = self.encoder.zs(batch['state'])
            next_zs = self.encoder.zs(batch['next_state'])
            sub_goal_zs = self.encoder.zs(batch['sub_goal'])
            phi_zs = self.encoder.phi(zs, sub_goal_zs)
            phi_next_zs = self.encoder.phi(next_zs, sub_goal_zs)

            v1, v2 = self.value(zs, phi_zs)
            nv1, nv2 = self.value(next_zs, phi_next_zs)
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v

            exp_a = torch.exp(adv * self.low_alpha).clamp(max=100.0).squeeze(-1)

            # Compute the goal representations of the subgoals.
            # Currently not learning the goal_rep
            # goal_reps = self.encoder.phi(zs, sub_goal_zs)
        
        dist = self.low_actor(zs, phi_zs, goal_encoded=True)
        log_prob = dist.log_prob(batch['action'])

        actor_loss = -(exp_a * log_prob).mean()
        
        #Optimier
        self.low_actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.low_actor_optimizer.step()

        actor_info = {
            'train/Lactor_loss': actor_loss.item(),
            'train/Lactor_adv': adv.mean().item(),
            'train/Lactor_bclogprob': log_prob.mean().item(),
            'train/Lactor_mse': torch.mean((dist.mean - batch['action']) ** 2).item(),
        } 

        return actor_info

    def train_high_actor(self, batch):
        """Compute the high-level actor loss."""
        with torch.no_grad():
            zs = self.encoder.zs(batch['state'])
            actor_goal_zs = self.encoder.zs(batch['actor_goal'])
            sub_goal_zs = self.encoder.zs(batch['sub_goal'])

            phi_zs = self.encoder.phi(zs, actor_goal_zs)
            phi_sub_zs = self.encoder.phi(sub_goal_zs, actor_goal_zs)
            
            # No gradients into target
            target = self.encoder.phi(zs, sub_goal_zs)

            v1, v2 = self.value(zs, phi_zs)
            nv1, nv2 = self.value(sub_goal_zs, phi_sub_zs)
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v

            exp_a = torch.exp(adv * self.high_alpha).clamp(max=100.0).squeeze(-1)

        dist = self.high_actor(zs, actor_goal_zs)
        
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()
        
        #Optimier
        self.high_actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.high_actor_optimizer.step()

        return {
            'train/Hactor_loss': actor_loss.item(),
            'train/Hactor_adv': adv.mean().item(),
            'train/Hactor_bclogprob': log_prob.mean().item(),
            'train/Hactor_mse': torch.mean((dist.mean - target) ** 2).item(),
            'train/Hactor_std': dist.base_dist.scale.mean().item(),
        }

    def select_action(self, state, goal, temperature=1.0):
        with torch.no_grad():
            if self.pixel_obs:
                state = torch.tensor(state.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)
                goal = torch.tensor(goal.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
                goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float, device=self.device)
            
            zs = self.encoder.zs(state)
            goal_zs = self.encoder.zs(goal)

            high_dist = self.high_actor(zs, goal_zs, temperature=temperature)
            goal_reps = high_dist.sample()
            if self.lengthnorm_fn:
                goal_reps = LengthNorm(goal_reps) 

            low_dist = self.low_actor(zs, goal_reps, goal_encoded=True, temperature=temperature)
            action = low_dist.sample()

            return int(action.argmax()) if self.discrete else action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action


    def target_update(self):
        
        # value update
        for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Encoder Update
        for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def train(self):
        metrics = {}
        self.training_steps += 1

        if self.hp.batched_buffer and self.training_steps % 25000 == 0:
            print("Agent: Updating Dataset")
            self.update_dataset()
        
        if (self.training_steps-1) % self.target_update_freq == 0:
            
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            if not(self.total_train_steps < self.training_steps and self.pause_encoder):
                for _ in range(self.target_update_freq):
                    if self.history == 0:
                        batch = self.replay_buffer.sample(horizon=self.enc_horizon, include_intermediate=True, gc_negative=False)
                    else:
                        batch = self.replay_buffer.sample_test(horizon=self.enc_horizon, include_intermediate=True, gc_negative=False)

                    batch['state'], batch['next_state'] = maybe_augment_state(batch['state'], batch['next_state'], self.pixel_obs, self.pixel_augs)
                    if self.enc_version == 'MrHi':
                        self.encoder_metric = self.train_encoder_MrHi(batch)
                    elif self.enc_version == 'MrHiAc':
                        self.encoder_metric = self.train_encoder_MrHiAc(batch)
                    elif self.enc_version == 'MrHiAcHd':
                        self.encoder_metric = self.train_encoder_MrHiAcHd(batch)
                    else:
                        self.encoder_metric = self.train_encoder(batch)
        
        metrics.update(self.encoder_metric)
        # self.target_update()
        
        if self.history == 0:
            batch = self.replay_buffer.sample(gc_negative=self.gc_negative, horizon=self.Q_horizon, include_intermediate=False)
        else:
            batch = self.replay_buffer.sample_test(gc_negative=self.gc_negative, horizon=self.Q_horizon, include_intermediate=False)

        batch['state'], batch['next_state'] = maybe_augment_state(batch['state'], batch['next_state'], self.pixel_obs, self.pixel_augs)
        batch['n-reward'], batch['term_discount'] = multi_step_reward(batch['reward'], batch['not_done'], self.discount)
        
        # train_critic
        value_metrics = self.train_critic(batch)
        metrics.update(value_metrics)        
        
        # train_actor
        
        low_actor_metrics = self.train_low_actor(batch)
        metrics.update(low_actor_metrics)

        high_actor_metrics = self.train_high_actor(batch)
        metrics.update(high_actor_metrics)
        
        return metrics


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float):
    ms_reward = 0
    scale = 1
    for i in range(reward.shape[1]):
        ms_reward += scale * reward[:,i]
        scale *= discount * not_done[:,i]
    
    return ms_reward, scale


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs and random.random() > 0.5:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


# Random shift.
def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)

# Added
def expectile_loss(adv, diff, expectile):
    """Compute the expectile loss."""
    weight = torch.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def LengthNorm(x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        dim = x.shape[-1]
        return x / norm * torch.sqrt(torch.tensor(dim, dtype=x.dtype, device=x.device))

def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()

# # Factory map
# WORLD_MODELS = {
#     "V2": WorldModel2H,
#     "V2R": WorldModelV2R, 
#     # "WM2": WorldModel2,    # Agent2
#     # "WM3": WorldModel3,    # Agent3
# }

# def make_world_model(name, zs_dim, action_dim, state_encoder, 
#                      goal_rep_fn, sigmoid_fn, lengthnorm_fn):
    
#     if name not in WORLD_MODELS:
#         raise ValueError(f"Unknown world model: {name}")
    
#     return WORLD_MODELS[name](
#         zs_dim=zs_dim,
#         action_dim=action_dim,
#         state_encoder=state_encoder,
#         goal_rep_fn=goal_rep_fn,
#         sigmoid=sigmoid_fn,
#         ln=lengthnorm_fn
#         )

class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins


    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot


    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)


    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)
    
# if __name__ == "__main__":
#     """
#     obs_shape: tuple, 
#     action_dim: int, 
#     max_action: float, 
#     pixel_obs: bool, 
#     discrete: bool,
#     device: torch.device, 
#     history: int=1, 
#     hp: Dict={}
#     """

#     state_dim = (3, 64, 64)
#     action_dim = 5
#     max_action = 1.0 
#     pixel_obs = True
#     discrete = False
#     device = torch.device('cuda')
#     history = 1
#     hp = {}

#     agent = Agent(state_dim, action_dim, max_action, pixel_obs, discrete, device, history, hp)
