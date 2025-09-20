import numpy as np
import torch 
from gymnasium.spaces import Box
import ogbench 
import gymnasium
import collections 
import gc

def toTensor(array, device):
    if len(array.shape) > 2:
        # pixel_obs
        d_type=torch.uint8
    else:
        d_type=torch.float32

    return torch.tensor(array, dtype=d_type, device=device)
    
def load_dataset(dataset_path, ob_dtype=np.float32, 
                 action_dtype=np.float32, compact_dataset=False,
                 storage_cpu=False):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
    """
    with np.load(dataset_path, mmap_mode='r') as file:
        dataset = dict()
        for k in ['observations', 'actions', 'terminals']:
            if k == 'observations':
                dtype = ob_dtype
            elif k == 'actions':
                dtype = action_dtype
            else:
                dtype = np.float32
            dataset[k] = file[k][...].astype(dtype, copy=False)
    
    if 'visual' in dataset_path:
        dataset['observations'] = dataset['observations'].transpose(0,3,1,2)
    
    if compact_dataset:

        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
        
        max_size = dataset['observations'].shape[0]
        state_shape = dataset['observations'].shape[1:]
        action_shape = dataset['actions'].shape[1:]
        # Store obs on GPU if they are sufficient small.
        memory, _ = torch.cuda.mem_get_info()
        obs_space = np.prod((max_size, * state_shape), dtype=np.int64) * 1 if ob_dtype == np.uint8 else 4
        ard_space = max_size * (action_shape[0] + 2) * 4
        if storage_cpu==False and obs_space + ard_space < memory:
            storage_device = torch.device('cuda')
        else:
            storage_device = torch.device('cpu')    
        print(f'Storage_device {storage_device}')

        dataset= {k:toTensor(v, storage_device) for k, v in dataset.items()}
    else:

        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        dataset['next_observations'] = dataset['observations'][next_ob_mask]
        dataset['observations'] = dataset['observations'][ob_mask]
        dataset['actions'] = dataset['actions'][ob_mask]
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

        # N X T X H X W X C
        
        terminals  = np.where(dataset['terminals'] == 1)[0]+1
        num_trajectory = len(terminals)
        traj_length = terminals[0]
        for k in dataset:
            dataset[k] = dataset[k].reshape(num_trajectory, traj_length, *dataset[k][0].shape)
        
    return dataset

class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    
    env = ogbench.make_env_and_datasets(dataset_name=env_name, env_only=True)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()
    

    return env


class CustomDataloader:
    def __init__(self, dataset_path, num_chunk=2, 
                         ob_dtype=np.float32, action_dtype=np.float32):
        """Generator that loads OGBench dataset in chunks."""
        # assert 'visual' in dataset_path # only for visual dataset
        self.ob_dtype = ob_dtype 
        self.action_dtype = action_dtype

        with np.load(dataset_path, mmap_mode='r') as file:
            self.obs = file['observations']
            self.act = file['actions']
            self.terminals = file['terminals']
            print(f"Observation Shape : {self.obs.shape}")
            if 'visual' in dataset_path:
                self.transpose_axes = (0, 3, 1, 2)
            else:
                self.transpose_axes = None

            self.end_idxs = np.where(self.terminals == 1)[0] + 1
            self.start_idxs = np.concatenate([[0], self.end_idxs[:-1]])

            self.total_size = len(self.start_idxs)

            self.num_chunk = num_chunk
            self.chunk_size = self.total_size//num_chunk
        
    def load_dataset_chunked(self, pos):
    
    
        # for pos in range(0, total_size, chunk_size):
        pos = pos * self.chunk_size
        start = self.start_idxs[pos] 
        end = self.end_idxs[min(pos + self.chunk_size, self.total_size-1)]
        print([start, end])
        obs_chunk = self.obs[start:end].astype(self.ob_dtype, copy=False)
        act_chunk = self.act[start:end].astype(self.action_dtype, copy=False)
        term_chunk = self.terminals[start:end].astype(np.float32, copy=False)

        if self.transpose_axes:
            obs_chunk = obs_chunk.transpose(*self.transpose_axes)

        # Recompute compact logic for this chunk
        valids = 1.0 - term_chunk
        next_term_chunk = np.concatenate([term_chunk[1:], [1.0]])
        new_terms = np.minimum(term_chunk + next_term_chunk, 1.0).astype(np.float32)

        storage_device = torch.device('cpu')
        
        # Chunk as numpy array 
        chunk = {
            'observations': obs_chunk,
            'actions': act_chunk,
            'terminals': new_terms,
            'valids': valids,
        }
        return chunk

def chunk2cuda(chunk, ob_dtype=torch.float32, act_dtype=torch.float32):

    for k, v in chunk.items():
        if k == 'observations':
            chunk[k] = torch.tensor(v, dtype=ob_dtype, device='cuda')
        elif k == 'actions':
            chunk[k] = torch.tensor(v, dtype=act_dtype, device='cuda')
        else:
            chunk[k] = torch.tensor(v, dtype=torch.float32, device='cuda')
    return chunk

# if __name__ == "__main__":
#     import time
#     # dataset_path='/home/toolkit/snowrepo/data/visual-antmaze-giant-navigate-v0.npz'
#     # dataset_path='/home/toolkit/snowrepo/data/visual-antmaze-giant-navigate-v0.npz'
#     dataset_path='/home/toolkit/snowrepo/data/visual-humanoidmaze-medium-navigate-v0.npz'
#     a = time.time()
#     ob_dtype = np.uint8 if 'visual' in dataset_path else np.float32
#     # print(ob_type)
    
#     Dloader = CustomDataloader(dataset_path, ob_dtype=ob_dtype, num_chunk=3)
#     print(f"{time.time() - a}")
#     for num, pos in enumerate(range(3)):
#         # torch.cuda.empty_cache()
#         # gc.collect()
        
#         chunk = Dloader.load_dataset_chunked(pos = pos)

#         # chunk = {k: v.to('cuda') for k, v in chunk.items()}
#         chunk = chunk2cuda(chunk, torch.uint8)
#         # device = torch.device("cuda:0")
#         # allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
#         # reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
#         # print(f"GPU 0 - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
#         # print(f'chunked_{num}')
#         # print(f"{time.time() - a}")
#         # del chunk
        

        
    