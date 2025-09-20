from ogb_utils import load_dataset
import numpy as np
import torch
import random 
from dataclasses import dataclass

def toTensor(array, d_type=torch.float, device=torch.device('cuda')):
    return torch.tensor(array, dtype=d_type, device=device)

def toDevice(tensor):
    if not tensor.is_cuda:
        return  tensor.to(torch.device('cuda'))
    return tensor

@dataclass
class Prob:
    cur: float
    rand: float
    traj: float

class ReplayBuffer():
    """
    Updated Version
    Intermediate horizon has goals included in it
    At this point the goals are sampled with geometric mean
    Suggested improvements
        1. Actor and Values goals 
        2. Valid idx with asset to check
        3. Uniform sampling 
        4. Terminal and Valids need not be in tensor
    """
    def __init__(self, batch_size, gamma=0.99, 
                value_current_p=0.2, actor_current_p=0.0,
                value_random_p=0.3, actor_random_p=0.0,
                value_traj_p=0.5, actor_traj_p=1.0, 
                weight=None, 
                storage_cpu=False, stack=0, sub_goal=25,
                comp_return=True, act_chunk=1, hd=False, sg=False):
        super(ReplayBuffer, self).__init__()
        self.batch_size = batch_size
        self.weight = weight
        self.gamma = gamma
        self.v_prob = Prob(value_current_p, value_random_p, value_traj_p)
        self.a_prob = Prob(actor_current_p, actor_random_p, actor_traj_p)
        self.storage_cpu = storage_cpu
        self.stack=stack
        self.sub_goal=sub_goal 
        self.comp_return = comp_return 
        self.act_chunk=act_chunk
        self.hd = hd # High Dynamics
        self.sg = sg
        # self.offset = []
    
    def load_ogbench(self, dataset_path, device='cuda'):
       
        if "visual" in dataset_path:
            self.obs_type = np.uint8
            self.normalize = 255.0
            self.pixel_obs = True
        else:
            self.obs_type = np.float32
            self.normalize = 1.0
            self.pixel_obs = False
        
        self.device = device

        dataset = load_dataset(dataset_path, self.obs_type, compact_dataset=True, storage_cpu=self.storage_cpu)
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.not_done = dataset['valids']
        self.valid_idxs = np.nonzero(dataset['valids'].cpu().numpy() > 0)[0]
        
        self.max_size = self.state.shape[0]
        self.state_shape = self.state.shape[1:]
      # Convert to list so it's mutable
        if self.stack > 0:
            self.state_shape = list(self.state.shape[1:])
            self.state_shape[0] *= self.stack  # Multiply channel dimension
            self.state_shape = tuple(self.state_shape)

        self.action_shape = self.action.shape[1:]
        self.storage_device = True if self.state.is_cuda else False
        
        self.traj_length = int(torch.diff(torch.where(self.not_done == 0)[0]).float().mean())
        self.num_traj = int(self.not_done.shape[0] // self.traj_length )
        self.size = np.prod(dataset['terminals'].shape) 
        if self.stack > 0:
            self.valid_idxs = self.valid_idxs.reshape(self.num_traj, -1)[:,self.stack-1:-(self.stack-1)].reshape(-1)
        # ASSERT valid index

    def sample_random_idx(self, num_samples):
        return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_samples)]

    def sample(self, 
        gc_negative=True, 
        horizon=1, 
        include_intermediate=False):

        batch_size = self.batch_size
        # sample index
        tid = np.random.randint(0, self.num_traj, batch_size).reshape(-1, 1)
        if self.hd:
            max_start = self.traj_length - horizon - 2 - (2*self.act_chunk) + 1
        else:
            max_start = self.traj_length - horizon - 2 -self.act_chunk + 1
        tpos = np.random.randint(1, max_start, batch_size)
        
        # Generate horizon indices
        local_ind = tpos[:, None] + np.arange(horizon + 1)  # to handle the next state-> 
        local_ind = np.clip(local_ind, 0, self.traj_length - 2) # most not required because its done with traj-horizon
        ind = self.traj_length * tid + local_ind
        # assert self.not_done[ind].all(), f"Sampled invalid indices from buffer! Check trajectory boundaries."
        if self.act_chunk > 1:
            act_ind = tpos[:, None] + np.arange(horizon+self.act_chunk+1)
            act_ind = self.traj_length * tid + act_ind
            action = self.action[act_ind]
            # Next+5 and Subgoals

        else:
            action = self.action[ind]
        
        min_max_reward = [-1,0] if gc_negative else [0, 1]

        if include_intermediate:
            
            next_state_extra = ind[:, :horizon] + self.act_chunk

            p_random = 0.0
            # Geometric sampling
            if random.random() < 0.5:
                offset = np.random.geometric(p=1 - self.gamma, size=batch_size) 
                offset = offset * np.where(np.random.rand(batch_size)< self.v_prob.cur , 0, 1)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2) # goal from
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
                # self.offset += list(np.minimum(offset, self.traj_length - 2))
            else:
                distances = np.random.rand(batch_size)  # Uniform [0, 1)
                offset = np.round(
                    local_ind[:, 0] * distances + self.traj_length * (1 - distances)
                ).astype(int)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(offset, self.traj_length - 2) # goal from
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
                # self.offset += list(np.minimum(offset-local_ind[:,0], self.traj_length - 2))
            not_done = toTensor(np.where(ind == goal_pos[:, None], 0, 1)[:, :horizon]) # not done
            if self.comp_return:
                reward = toTensor(self.gamma**(goal_pos[:, None] - ind[:, :horizon]).clip(0, self.traj_length))
            else:
                reward = toTensor(np.where(ind == goal_pos[:, None], 0, -1)[:, :horizon])

            concat_ind = np.hstack((ind, goal_pos[:, None])) # concat mid goals
            
            if self.hd and self.act_chunk>1:
                # High Dynamics
                next_state_extra_pos = ind[:, :horizon] + self.act_chunk
                # assert self.not_done[next_state_extra_pos].all(), "Sampled invalid goal indices (goal_pos)"
                next_state_extra = self.state[next_state_extra_pos].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
                
            else:
                next_state_extra = None
            if self.sg:
                # Sub goal
                sub_goal_pos = np.minimum(ind[:, :horizon] + self.sub_goal, goal_pos[:, None])
                # assert self.not_done[sub_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
                sub_goal = self.state[sub_goal_pos].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            else:
                sub_goal = None
            
            both_state = self.state[concat_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = both_state[:,:-2]       # State: (batch_size, horizon, *state_dim)
            next_state = both_state[:,1:-1]   # Next state: (batch_size, horizon, *state_dim)
            goal = both_state[:,-1]
            action = action[:,:-1, :]         # Action: (batch_size, horizon, action_dim)
            
            if self.storage_device:
                return dict(state=state, action=action, next_state=next_state, goal=goal, 
                            not_done=not_done, reward=reward, sub_goal=sub_goal, next_state_extra=next_state_extra)
            else:
                return dict(state=toDevice(state), action=toDevice(action), next_state=toDevice(next_state), 
                            goal=toDevice(goal), not_done=not_done, reward=reward, 
                            sub_goal=toDevice(sub_goal), next_state_extra=toDevice(next_state_extra))

        else:
            value_goal_pos = self.value_goal_idx(local_ind[:, 0], tid, p_current=self.v_prob.cur, 
                                                p_traj=self.v_prob.traj, p_random=self.v_prob.rand)
            # assert self.not_done[value_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            
            actor_goal_pos, actor_subgoal_pos = self.actor_goal_idx(local_ind[:, 0], tid, p_random=self.a_prob.rand, 
                                                                    p_traj=self.a_prob.traj)
            # assert self.not_done[actor_subgoal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            # assert self.not_done[actor_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            
            weight = None

            not_done = toTensor(np.where(ind == value_goal_pos[:, None], 0, 1)[:, :horizon]).unsqueeze(-1)
            reward = toTensor(np.where(ind[:, :-1] == value_goal_pos[:, None], min_max_reward[1], min_max_reward[0])[:, :horizon]).unsqueeze(-1)
            
            stacked_ind = np.stack((ind[:, 0], ind[:, -1], value_goal_pos, actor_goal_pos, actor_subgoal_pos), 1)
            all_state = self.state[stacked_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = all_state[:,0]       # State: (batch_size, *state_dim)
            next_state = all_state[:,1]   # Next state: (batch_size, *state_dim)
            action = action[:,0]
            value_goal = all_state[:,2]
            actor_goal = all_state[:,3]
            actor_subgoal = all_state[:,4]

            if self.storage_device:
                return dict(state=state, action=action, next_state=next_state, value_goal=toDevice(value_goal),
                            actor_goal=toDevice(actor_goal), not_done=not_done, reward=reward, weight=weight,
                            sub_goal=actor_subgoal)
            else:
                return dict(state=toDevice(state), action=toDevice(action), next_state=toDevice(next_state), 
                            value_goal=toDevice(value_goal), actor_goal=toDevice(actor_goal), not_done=not_done, 
                            reward=reward, weight=weight, sub_goal=toDevice(actor_subgoal))
    
    def value_goal_idx(self, idxs, tid, p_current, p_traj, p_random):

        value_random_idx = self.sample_random_idx(self.batch_size)
        value_offset = np.random.geometric(p=1 - self.gamma, size=self.batch_size) 
        value_offset = value_offset * np.where(np.random.rand(self.batch_size)<(p_current/(1-p_random)), 0, 1)
        value_goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(idxs + value_offset, self.traj_length - 2) 

        # Choice between random and value_goal
        if p_random > 0:
            goal_pos = np.where(np.random.rand(self.batch_size) < p_random, value_random_idx, value_goal_pos)
        else:
            goal_pos = value_goal_pos
        return goal_pos

    def actor_goal_idx(self, idxs, tid, p_random, p_traj):
        # random_goals and Sub goals 
        random_goal_pos = self.sample_random_idx(self.batch_size)
        random_subgoal_pos = self.traj_length * tid.reshape(-1) + np.minimum(idxs + self.sub_goal, self.traj_length - 2)

        # Actor goal
        distances = np.random.rand(self.batch_size)  # Uniform [0, 1)
        actor_offset = np.round(
            idxs * distances + self.traj_length * (1 - distances)
        ).astype(int) 
        actor_goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(actor_offset, self.traj_length - 2)
        
        # Sub Goal 
        actor_subgoal_pos = self.traj_length * tid.reshape(-1) + np.minimum(idxs + self.sub_goal, self.traj_length - 2)
        actor_subgoal_pos = np.minimum(actor_subgoal_pos, actor_goal_pos)

        # Choice between random and actor goal
        pick_random = np.random.rand(self.batch_size) < self.a_prob.rand
        goal_pos = np.where(pick_random, random_goal_pos, actor_goal_pos)
        subgoal_pos = np.where(pick_random, random_subgoal_pos, actor_subgoal_pos)
        return goal_pos, subgoal_pos

    def sample_test(self, 
        gc_negative=True, 
        horizon=1, 
        include_intermediate=False
        ):

        batch_size = self.batch_size
        # sample index
        tid = np.random.randint(0, self.num_traj, batch_size).reshape(-1, 1)
        if self.hd:
            max_start = self.traj_length - horizon - 2 - (2*self.act_chunk) + 1
        else:
            max_start = self.traj_length - horizon - 2 -self.act_chunk + 1
        tpos = np.random.randint(self.stack, max_start, batch_size)
        
        # Generate horizon indices
        local_ind = tpos[:, None] + np.arange(horizon + 1)  # to handle the next state-> 
        local_ind = np.clip(local_ind, 0, self.traj_length - 2) # most not required because its done with traj-horizon
        ind = self.traj_length * tid + local_ind

        stack_offset = np.arange(-self.stack + 1, 1).reshape(1, 1, -1)  # (1,1,stack)
        stack_indices = self.traj_length * tid[..., None] + (local_ind[..., None] + stack_offset).clip(0, self.traj_length - 2)
        # assert self.not_done[stack_indices].all(), f"Sampled invalid indices from buffer! Check trajectory boundaries."
        # assert self.not_done[ind].all(), f"Sampled invalid indices from buffer! Check trajectory boundaries."
        if self.act_chunk > 1:
            act_ind = tpos[:, None] + np.arange(horizon+self.act_chunk+1)
            act_ind = self.traj_length * tid + act_ind
            action = self.action[act_ind]
        else:
            action = self.action[ind]
        
        min_max_reward = [-1,0] if gc_negative else [0, 1]

        if include_intermediate:
            p_random = 0.0
            # Geometric sampling
            if random.random() < 0.5:
                offset = np.random.geometric(p=1 - self.gamma, size=batch_size) 
                offset = offset * np.where(np.random.rand(batch_size)< self.v_prob.cur/(1-p_random) , 0, 1)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2)
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            else:
                distances = np.random.rand(batch_size)  # Uniform [0, 1)
                offset = np.round(
                    local_ind[:, 0] * distances + self.traj_length * (1 - distances)
                ).astype(int)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2)
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            # self.offset += list(offset)

            not_done = toTensor(np.where(ind == goal_pos[:, None], 0, 1)[:, :horizon]) # not done
            if self.comp_return:
                reward = toTensor(self.gamma**(goal_pos[:, None] - ind[:, :horizon]).clip(0, self.traj_length))
            else:
                reward = toTensor(np.where(ind == goal_pos[:, None], 0, -1)[:, :horizon])

            if self.hd and self.act_chunk>1:
                # High Dynamics
                next_state_extra_pos = ind[:, :horizon] + self.act_chunk
                stack_next_extra_pos = next_state_extra_pos[:, :, None] + np.arange(-self.stack + 1, 1).reshape(1, 1, -1)
                assert self.not_done[stack_next_extra_pos].all(), "Sampled invalid goal indices (goal_pos)"
                next_state_extra = self.state[stack_next_extra_pos].reshape(self.batch_size,horizon,*self.state_shape).type(torch.float)
            else:
                next_state_extra = None
            

            stacked_state = self.state[stack_indices].type(torch.float)  # shape: (B*(horizon+1)*stack, C, H, W)
            stacked_state = stacked_state.reshape(batch_size, horizon + 1, *self.state_shape)  # (B, H+1, stack, C, H, W)
            # stacked_state = stacked_state.transpose(0, 1, 3, 2, 4, 5).reshape(batch_size, horizon + 1, -1, *self.state_shape[1:])  # (B, H+1, stack*C, H, W)
            state      = stacked_state[:, :-1]  # (B, horizon, C*stack, H, W)
            next_state = stacked_state[:, 1:] 
            
            goal_stack = np.arange(-self.stack + 1, 1).reshape(1, -1)
            goal_stack = goal_pos[:, None] + goal_stack

            if self.sg:
                # Sub goal
                sub_goal_pos = np.minimum(ind[:, :horizon] + self.sub_goal, goal_pos[:, None])
                sub_goal_stack = np.arange(-self.stack + 1, 1).reshape(1, 1, -1) + sub_goal_pos[..., None] 
                # assert self.not_done[sub_goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
                sub_goal = self.state[sub_goal_stack].type(torch.float)  # (B, stack, C, H, W)
                sub_goal = sub_goal.reshape(batch_size, horizon, *self.state_shape) 
            else:
                sub_goal = None
            
            # assert self.not_done[goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
            goal = self.state[goal_stack].type(torch.float)  # (B, stack, C, H, W)
            goal = goal.reshape(batch_size, *self.state_shape)  # (B, C*stack, H, W)

            action = action[:,:-1, :]         # Action: (batch_size, horizon, action_dim)
            
            if self.storage_device:
                return dict(state=state, action=action, next_state=next_state, goal=goal, not_done=not_done, reward=reward,
                            sub_goal=sub_goal, next_state_extra=next_state_extra)
            else:
                return dict(state=toDevice(state), action=toDevice(action), next_state=toDevice(next_state), goal=toDevice(goal), 
                            not_done=not_done, reward=reward, sub_goal=toDevice(sub_goal), next_state_extra=toDevice(next_state_extra))

        else:
            value_goal_pos = self.value_goal_idx(local_ind[:, 0], tid, p_current=self.v_prob.cur, 
                                                p_traj=self.v_prob.traj, p_random=self.v_prob.rand)
            # assert self.not_done[value_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            
            actor_goal_pos, actor_subgoal_pos = self.actor_goal_idx(local_ind[:, 0], tid, p_random=self.a_prob.rand, 
                                                                    p_traj=self.a_prob.traj)
            # assert self.not_done[actor_subgoal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            # assert self.not_done[actor_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            weight = None
            
            not_done = toTensor(np.where(ind == value_goal_pos[:, None], 0, 1)[:, :horizon]).unsqueeze(-1)
            reward = toTensor(np.where(ind[:, :-1] == value_goal_pos[:, None], min_max_reward[1], min_max_reward[0])[:, :horizon]).unsqueeze(-1)
            
            state = self.state[stack_indices[:, 0]].reshape(batch_size, *self.state_shape).type(torch.float)  # shape: (B*(horizon+1)*stack, C, H, W)
            next_state = self.state[stack_indices[:, -1]].reshape(batch_size, *self.state_shape).type(torch.float)

            goal_stack = np.arange(-self.stack + 1, 1).reshape(1, -1)
            value_goal_stack = value_goal_pos[:, None] + goal_stack
            actor_goal_stack = actor_goal_pos[:, None] + goal_stack
            actor_subgoal_stack = actor_subgoal_pos[:, None] + goal_stack
            # assert self.not_done[value_goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
            # assert self.not_done[actor_goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
            # assert self.not_done[actor_subgoal_stack].all(), "Sampled invalid goal indices (goal_pos)"

            value_goal = self.state[value_goal_stack].type(torch.float)  # (B, stack, C, H, W)
            value_goal = value_goal.reshape(batch_size, *self.state_shape)  # (B, C*stack, H, W)

            actor_goal = self.state[actor_goal_stack].type(torch.float)  # (B, stack, C, H, W)
            actor_goal = actor_goal.reshape(batch_size, *self.state_shape)

            actor_subgoal = self.state[actor_subgoal_stack].type(torch.float)  # (B, stack, C, H, W)
            actor_subgoal = actor_goal.reshape(batch_size, *self.state_shape)
            
            action = action[:,0]

            if self.storage_device:
                return dict(state=state, action=action, next_state=next_state, value_goal=toDevice(value_goal),
                            actor_goal=toDevice(actor_goal), not_done=not_done, reward=reward, weight=weight,
                            sub_goal=toDevice(actor_subgoal))
            else:
                return dict(state=toDevice(state), action=toDevice(action), next_state=toDevice(next_state), 
                            value_goal=toDevice(value_goal), actor_goal=toDevice(actor_goal), not_done=not_done, 
                            reward=reward, weight=weight, sub_goal=actor_subgoal)

"""
if __name__ == "__main__":
    # dataset_path = '/home/toolkit/snowrepo/data/visual-humanoidmaze-medium-navigate-v0-val.npz'
    dataset_path='/home/toolkit/snowrepo/data/visual-antmaze-giant-navigate-v0-val.npz'
    # dataset_path='/home/toolkit/snowrepo/data/antmaze-giant-navigate-v0.npz'
    batch_size = 256
    buffer = ReplayBuffer(batch_size=batch_size, gamma=0.995, weight=None, stack=3, sub_goal=25, comp_return=False, hd=False, act_chunk=5, sg=True)
    buffer.load_ogbench(dataset_path=dataset_path)

    for i in range(10000):
        out = buffer.sample_test(horizon=5, include_intermediate=True)
        out = buffer.sample_test(horizon=3, include_intermediate=False)

    # buffer = ReplayBuffer(batch_size=batch_size, gamma=0.995, weight=None, stack=0, sub_goal=25, comp_return=False, hd=False, act_chunk=5, sg=True)
    # buffer.load_ogbench(dataset_path=dataset_path)
    # for i in range(10000):
    #     out = buffer.sample(horizon=5, include_intermediate=True)
    #     out = buffer.sample(horizon=3, include_intermediate=False)
    
    # import matplotlib.pyplot as plt
    # plt.hist(buffer.offset, bins=1000, edgecolor='black')  # bins controls resolution
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Data")
    # plt.savefig('image.png')
    check = 1
"""