try:
    from common.ogb_utils import load_dataset, CustomDataloader, chunk2cuda
    from common.buffer import ReplayBuffer
    
except:
    from ogb_utils import load_dataset, CustomDataloader, chunk2cuda
    from buffer import ReplayBuffer
import numpy as np
import torch
import random 
import gc 

class ReplayBufferBatch(ReplayBuffer):
    def __init__(self, *args, chunks, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks = chunks 
        self.chunk_id = 0
    
    def load_ogbench(self, dataset_path, device='cuda'):
        if "visual" in dataset_path:
            self.obs_type = np.uint8
            self.obs_type_cuda = torch.uint8
            self.normalize = 255.0
            self.pixel_obs = True
        else:
            self.obs_type = np.float32
            self.obs_type_cuda = torch.float32
            self.normalize = 1.0
            self.pixel_obs = False
        
        self.device = device

        self.Dloader = CustomDataloader(dataset_path, ob_dtype=self.obs_type, num_chunk=self.chunks)
        dataset = self.Dloader.load_dataset_chunked(pos = (self.chunk_id % self.chunks))
        dataset = chunk2cuda(dataset, ob_dtype=self.obs_type_cuda)
        # dataset = load_dataset(dataset_path, self.obs_type, compact_dataset=True, storage_cpu=self.storage_cpu)
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
    
    def update_dataset(self):
        print('[Info] Updating the New chunk')
        del self.state, self.action, self.not_done, # dataset
        torch.cuda.empty_cache()
        gc.collect()
        self.chunk_id += 1
        dataset = self.Dloader.load_dataset_chunked(pos = (self.chunk_id % self.chunks))
        dataset = chunk2cuda(dataset, ob_dtype=self.obs_type_cuda)
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
    
