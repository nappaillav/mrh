from ogb_utils import load_dataset
import numpy as np
import torch
import random 

def toTensor(array, d_type=torch.float, device=torch.device('cuda')):
    return torch.tensor(array, dtype=d_type, device=device)

def toDevice(tensor):
    if not tensor.is_cuda:
        return  tensor.to(torch.device('cuda'))
    return tensor

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
    def __init__(self, batch_size, gamma=0.99, value_goal_p=0.25, actor_goal_p=0.0, weight=None, 
                 storage_cpu=False, stack=0, sub_goal=25, comp_return=True, act_chunk=1):
        super(ReplayBuffer, self).__init__()
        self.batch_size = batch_size
        self.weight = weight
        self.gamma = gamma
        self.value_goal_p = value_goal_p
        self.actor_goal_p = actor_goal_p
        self.storage_cpu = storage_cpu
        self.stack=stack
        self.sub_goal=sub_goal 
        self.comp_return = comp_return
        self.act_chunk = act_chunk

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
        # ASSERT valid index

    def sample(self, 
        gc_negative=True, 
        horizon=1, 
        include_intermediate=False):

        batch_size = self.batch_size
        # sample index
        tid = np.random.randint(0, self.num_traj, batch_size).reshape(-1, 1)
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
        else:
            action = self.action[ind]
        
        min_max_reward = [-1,0] if gc_negative else [0, 1]

        if include_intermediate:
            
            # Geometric sampling
            if random.random() < 0.5:
                offset = np.random.geometric(p=1 - self.gamma, size=batch_size) 
                offset = offset * np.where(np.random.rand(batch_size)< self.value_goal_p, 0, 1)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2) # goal from
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            else:
                distances = np.random.rand(batch_size)  # Uniform [0, 1)
                offset = np.round(
                    local_ind[:, 0] * distances + self.traj_length * (1 - distances)
                ).astype(int)
                # offset = np.round(1 * distances + (self.traj_length-local_ind[:, 0]) * (1 - distances)).astype(int)
                # offset = offset * np.where(np.random.rand(batch_size)< self.value_goal_p, 0, 1)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(offset, self.traj_length - 2) # goal from
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            mid_pos = ((ind[:, 0] + goal_pos) / 2).astype('int')

            not_done = toTensor(np.where(ind == goal_pos[:, None], 0, 1)[:, :horizon]) # not done
            if self.comp_return:
                reward = toTensor(self.gamma**(goal_pos[:, None] - ind[:, :horizon]).clip(0, self.traj_length))
            else:
                reward = toTensor(np.where(ind == goal_pos[:, None], 0, -1)[:, :horizon])

            concat_ind = np.hstack((ind, goal_pos[:, None])) # concat mid goals

            both_state = self.state[concat_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = both_state[:,:-2]       # State: (batch_size, horizon, *state_dim)
            next_state = both_state[:,1:-1]   # Next state: (batch_size, horizon, *state_dim)
            goal = both_state[:,-1]
            action = action[:,:-1, :]         # Action: (batch_size, horizon, action_dim)
            
            if self.storage_device:
                return dict(state=state, action=action, next_state=next_state, goal=goal, not_done=not_done, reward=reward)
            else:
                return dict(state=toDevice(state), action=toDevice(action), next_state=toDevice(next_state), goal=toDevice(goal), 
                            not_done=not_done, reward=reward)

        else:
            value_offset = np.random.geometric(p=1 - self.gamma, size=batch_size) 
            value_offset = value_offset * np.where(np.random.rand(batch_size)<self.value_goal_p, 0, 1)
            value_goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + value_offset, self.traj_length - 2) 
            # assert self.not_done[value_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            distances = np.random.rand(batch_size)  # Uniform [0, 1)
            actor_offset = np.round(
                local_ind[:, 0] * distances + self.traj_length * (1 - distances)
            ).astype(int) 
            # actor_offset = np.round(1 * distances + (self.traj_length-local_ind[:, 0]) * (1 - distances)).astype(int)
            # actor_offset = actor_offset * np.where(np.random.rand(batch_size) < self.actor_goal_p, 0, 1)
            actor_goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(actor_offset, self.traj_length - 2)
            assert self.not_done[actor_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            
            # sub_goal 
            actor_subgoal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:,0] + self.sub_goal, self.traj_length - 2)
            actor_subgoal_pos = np.minimum(actor_subgoal_pos, actor_goal_pos)
            assert self.not_done[actor_subgoal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            # Weight calculation used by the policy 
            # if self.weight == 'exp':
            #     weight = self.gamma ** (value_goal_pos - ind[:, 0])
            # elif self.weight == 'linear':
            #     weight = 1 - ((value_goal_pos - ind[:, 0]) / (self.traj_length + 1e-8))
            # else:
            #     weight = np.ones_like(value_goal_pos, dtype=np.float32)
            # assert np.all(weight >= 0), "Negative weights detected!"
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

    def sample_test(self, 
        gc_negative=True, 
        horizon=1, 
        include_intermediate=False
        ):
        
        batch_size = self.batch_size
        # sample index
        tid = np.random.randint(0, self.num_traj, batch_size).reshape(-1, 1)
        max_start = self.traj_length - horizon - 2 - self.act_chunk + 1
        tpos = np.random.randint(self.stack, max_start, batch_size)
        
        # Generate horizon indices
        local_ind = tpos[:, None] + np.arange(horizon + 1)  # to handle the next state-> 
        local_ind = np.clip(local_ind, 0, self.traj_length - 2) # most not required because its done with traj-horizon
        ind = self.traj_length * tid + local_ind

        stack_offset = np.arange(-self.stack + 1, 1).reshape(1, 1, -1)  # (1,1,stack)
        stack_indices = self.traj_length * tid[..., None] + (local_ind[..., None] + stack_offset).clip(0, self.traj_length - 2)
        # stack_indices = stack_indices.reshape(-1) 
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
            
            # Geometric sampling
            if random.random() < 0.5:
                offset = np.random.geometric(p=1 - self.gamma, size=batch_size) 
                offset = offset * np.where(np.random.rand(batch_size)< self.value_goal_p, 0, 1)
                # goal_pos = np.minimum(local_ind[:, 0] + offset, self.traj_length - 2) # goal from
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2)
                # Add stacking 
                # goal_stack = np.arange(-self.stack + 1, 1).reshape(1, -1)
                # goal_offset_stack = (goal_pos[..., None] + goal_stack).clip(0, self.traj_length - 2)
                # goal_pos = self.traj_length * tid + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2) # (Bxstack)# goal from
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            else:
                distances = np.random.rand(batch_size)  # Uniform [0, 1)
                offset = np.round(
                    local_ind[:, 0] * distances + self.traj_length * (1 - distances)
                ).astype(int)
                # offset = np.round(1 * distances + (self.traj_length-local_ind[:, 0]) * (1 - distances)).astype(int)
                # offset = offset * np.where(np.random.rand(batch_size)< self.value_goal_p, 0, 1)
                # goal_stack = np.arange(-self.stack + 1, 1).reshape(1, -1)
                # goal_pos = (goal_pos[..., None] + goal_stack).clip(0, self.traj_length - 2)
                goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 2)
                # goal_pos = self.traj_length * tid + np.minimum(offset, self.traj_length - 2) # goal from
                # assert self.not_done[goal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            mid_pos = ((ind[:, 0] + goal_pos) / 2).astype('int')

            not_done = toTensor(np.where(ind == goal_pos[:, None], 0, 1)[:, :horizon]) # not done
            if self.comp_return:
                reward = toTensor(self.gamma**(goal_pos[:, None] - ind[:, :horizon]).clip(0, self.traj_length))
            else:
                reward = toTensor(np.where(ind == goal_pos[:, None], 0, -1)[:, :horizon])

            # concat_ind = np.hstack((ind, goal_pos[:, None])) # concat mid goals

            # both_state = self.state[concat_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            # state = both_state[:,:-2]       # State: (batch_size, horizon, *state_dim)
            # next_state = both_state[:,1:-1]   # Next state: (batch_size, horizon, *state_dim)
            # goal = both_state[:,-1]
            stacked_state = self.state[stack_indices].type(torch.float)  # shape: (B*(horizon+1)*stack, C, H, W)
            stacked_state = stacked_state.reshape(batch_size, horizon + 1, *self.state_shape)  # (B, H+1, stack, C, H, W)
            # stacked_state = stacked_state.transpose(0, 1, 3, 2, 4, 5).reshape(batch_size, horizon + 1, -1, *self.state_shape[1:])  # (B, H+1, stack*C, H, W)
            state      = stacked_state[:, :-1]  # (B, horizon, C*stack, H, W)
            next_state = stacked_state[:, 1:] 
            
            goal_stack = np.arange(-self.stack + 1, 1).reshape(1, -1)
            goal_stack = goal_pos[:, None] + goal_stack
            
            # assert self.not_done[goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
            goal = self.state[goal_stack].type(torch.float)  # (B, stack, C, H, W)
            goal = goal.reshape(batch_size, *self.state_shape)  # (B, C*stack, H, W)

            action = action[:,:-1, :]         # Action: (batch_size, horizon, action_dim)
            
            if self.storage_device:
                return dict(state=state, action=action, next_state=next_state, goal=goal, not_done=not_done, reward=reward)
            else:
                return dict(state=toDevice(state), action=toDevice(action), next_state=toDevice(next_state), goal=toDevice(goal), 
                            not_done=not_done, reward=reward)

        else:
            value_offset = np.random.geometric(p=1 - self.gamma, size=batch_size) 
            value_offset = value_offset * np.where(np.random.rand(batch_size)<self.value_goal_p, 0, 1)
            value_goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + value_offset, self.traj_length - 2) 
            # assert self.not_done[value_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            distances = np.random.rand(batch_size)  # Uniform [0, 1)
            actor_offset = np.round(
                local_ind[:, 0] * distances + self.traj_length * (1 - distances)
            ).astype(int) 
            # actor_offset = np.round(1 * distances + (self.traj_length-local_ind[:, 0]) * (1 - distances)).astype(int)
            # actor_offset = actor_offset * np.where(np.random.rand(batch_size) < self.actor_goal_p, 0, 1)
            actor_goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(actor_offset, self.traj_length - 2)
            assert self.not_done[actor_goal_pos].all(), "Sampled invalid goal indices (goal_pos)"
            
            # sub_goal
            actor_subgoal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:,0] + self.sub_goal, self.traj_length - 2)
            actor_subgoal_pos = np.minimum(actor_subgoal_pos, actor_goal_pos)
            assert self.not_done[actor_subgoal_pos].all(), "Sampled invalid goal indices (goal_pos)"

            # Weight calculation used by the policy 
            if self.weight == 'exp':
                weight = self.gamma ** (value_goal_pos - ind[:, 0])
            elif self.weight == 'linear':
                weight = 1 - ((value_goal_pos - ind[:, 0]) / (self.traj_length + 1e-8))
            else:
                weight = np.ones_like(value_goal_pos, dtype=np.float32)
            assert np.all(weight >= 0), "Negative weights detected!"
            
            not_done = toTensor(np.where(ind == value_goal_pos[:, None], 0, 1)[:, :horizon]).unsqueeze(-1)
            reward = toTensor(np.where(ind[:, :-1] == value_goal_pos[:, None], min_max_reward[1], min_max_reward[0])[:, :horizon]).unsqueeze(-1)
            
            # stacked_ind = np.stack((ind[:, 0], ind[:, -1], value_goal_pos, actor_goal_pos), 1)
            # all_state = self.state[stacked_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            # state = all_state[:,0]       # State: (batch_size, *state_dim)
            # next_state = all_state[:,1]   # Next state: (batch_size, *state_dim)
            # action = action[:,0]
            # value_goal = all_state[:,2]
            # actor_goal = all_state[:,3]
            state = self.state[stack_indices[:, 0]].reshape(batch_size, *self.state_shape).type(torch.float)  # shape: (B*(horizon+1)*stack, C, H, W)
            next_state = self.state[stack_indices[:, -1]].reshape(batch_size, *self.state_shape).type(torch.float)

            goal_stack = np.arange(-self.stack + 1, 1).reshape(1, -1)
            value_goal_stack = value_goal_pos[:, None] + goal_stack
            actor_goal_stack = actor_goal_pos[:, None] + goal_stack
            actor_subgoal_stack = actor_subgoal_pos[:, None] + goal_stack
            # assert self.not_done[value_goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
            # assert self.not_done[actor_goal_stack].all(), "Sampled invalid goal indices (goal_pos)"
            assert self.not_done[actor_subgoal_stack].all(), "Sampled invalid goal indices (goal_pos)"

            value_goal = self.state[value_goal_stack].type(torch.float)  # (B, stack, C, H, W)
            value_goal = value_goal.reshape(batch_size, *self.state_shape)  # (B, C*stack, H, W)

            actor_goal = self.state[actor_goal_stack].type(torch.float)  # (B, stack, C, H, W)
            actor_goal = actor_goal.reshape(batch_size, *self.state_shape)

            actor_subgoal = self.state[actor_goal_stack].type(torch.float)  # (B, stack, C, H, W)
            actor_subgoal = actor_subgoal.reshape(batch_size, *self.state_shape)
            
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
    # dataset_path='/home/toolkit/snowrepo/data/visual-antmaze-giant-navigate-v0.npz'
    # dataset_path='/home/toolkit/snowrepo/data/antmaze-giant-navigate-v0-val.npz'
    dataset_path='/home/toolkit/snowrepo/data/visual-cube-single-play-v0.npz'
    batch_size = 256
    # buffer = ReplayBuffer(batch_size=batch_size, gamma=0.995, weight=None, stack=3, sub_goal=25, act_chunk=5)
    # buffer.load_ogbench(dataset_path=dataset_path)

    # for i in range(10000):
    #     out = buffer.sample_test(horizon=5, include_intermediate=True)
    #     out = buffer.sample_test(horizon=3, include_intermediate=False)

    buffer = ReplayBuffer(batch_size=batch_size, gamma=0.995, weight=None, stack=0, sub_goal=25)
    buffer.load_ogbench(dataset_path=dataset_path)
    for i in range(10000):
        # out = buffer.sample(horizon=5, include_intermediate=True)
        out = buffer.sample(horizon=1, include_intermediate=False)
    check = 1

"""