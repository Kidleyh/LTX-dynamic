import sys
# from ltx_distillation.utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os


# import LTX_2.ltx_core as ltx_core
# class TextDataset(Dataset):
#     def __init__(self, prompt_path, extended_prompt_path=None):
#         with open(prompt_path, encoding="utf-8") as f:
#             self.prompt_list = [line.rstrip() for line in f]

#         if extended_prompt_path is not None:
#             with open(extended_prompt_path, encoding="utf-8") as f:
#                 self.extended_prompt_list = [line.rstrip() for line in f]
#             assert len(self.extended_prompt_list) == len(self.prompt_list)
#         else:
#             self.extended_prompt_list = None

#     def __len__(self):
#         return len(self.prompt_list)

#     def __getitem__(self, idx):
#         batch = {
#             "prompts": self.prompt_list[idx],
#             "idx": idx,
#         }
#         if self.extended_prompt_list is not None:
#             batch["extended_prompts"] = self.extended_prompt_list[idx]
#         return batch


# class ODERegressionLMDBDataset(Dataset):
#     def __init__(self, data_path: str, max_pair: int = int(1e8)):
#         self.env = lmdb.open(data_path, readonly=True,
#                              lock=False, readahead=False, meminit=False)

#         self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
#         self.max_pair = max_pair

#     def __len__(self):
#         return min(self.latents_shape[0], self.max_pair)

#     def __getitem__(self, idx):
#         """
#         Outputs:
#             - prompts: List of Strings
#             - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
#         """
#         latents = retrieve_row_from_lmdb(
#             self.env,
#             "latents", np.float16, idx, shape=self.latents_shape[1:]
#         )

#         if len(latents.shape) == 4:
#             latents = latents[None, ...]

#         prompts = retrieve_row_from_lmdb(
#             self.env,
#             "prompts", str, idx
#         )
#         return {
#             "prompts": prompts,
#             "ode_latent": torch.tensor(latents, dtype=torch.float32)
#         }


class ODERegressionStateDictDataset(Dataset):
    def __init__(self, data_paths: list, max_pair: int = int(1e8)):
        # self.env = lmdb.open(data_path, readonly=True,
        #                      lock=False, readahead=False, meminit=False)

        all_states = []
        for data_path in data_paths:
            data_list = [i for i in os.listdir(data_path) if i.endswith(".pt")]
            all_states.extend([os.path.join(data_path, i) for i in data_list])
        self.all_states = all_states
        self.max_refetch = 5

    def __len__(self):
        return len(self.all_states)

    def _rand_another(self):
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))


    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        for _ in range(self.max_refetch + 1):
            try:
                train_info_dict = torch.load(self.all_states[idx], map_location=torch.device('cpu'), weights_only=False)
            except Exception as e:
                train_info_dict = None
                print(e, self.all_states[idx])
            # Broken images or random augmentations may cause the returned data
            # to be None
            if train_info_dict is None:
                idx = self._rand_another()
                continue
            return train_info_dict

        return train_info_dict

        # if len(latents.shape) == 4:
        #     latents = latents[None, ...]

        # prompts = retrieve_row_from_lmdb(
        #     self.env,
        #     "prompts", str, idx
        # )
        # return {
        #     "prompts": prompts,
        #     "ode_latent": torch.tensor(latents, dtype=torch.float32)
        # }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
    

def ltx_collate_fn(batch): 
    ltx_ode_data_dict = dict()
    # ode_sigma_index = [0,7,8,9]
    # ode_sigma_index = [0,6,7,9]
    # ode_sigma_index = [0,1,2,3]
    ode_sigma_index = [0,1,3,6] # [0,2,5,6]
    ltx_ode_data_dict["video_ode_latents"] = []
    ltx_ode_data_dict["video_denoise_mask"] = []
    ltx_ode_data_dict["video_positions"] = []
    ltx_ode_data_dict["video_clean_latents"] = []
    
    ltx_ode_data_dict["audio_ode_latents"] = []
    ltx_ode_data_dict["audio_denoise_mask"] = []
    ltx_ode_data_dict["audio_positions"] = []
    ltx_ode_data_dict["audio_clean_latents"] = []
    
    ltx_ode_data_dict["video_positive_context"] = []
    ltx_ode_data_dict["audio_positive_context"] = []
    
    for item in batch:
        ltx_ode_data_dict["video_ode_latents"].append(torch.cat([item["video_noisy_inputs"][i].latent for i in range(len(item["video_noisy_inputs"])) if i in ode_sigma_index], dim=0))
        ltx_ode_data_dict["video_denoise_mask"].append(torch.cat([latent_state.denoise_mask for latent_state in item["video_noisy_inputs"]], dim=0))
        ltx_ode_data_dict["video_positions"].append(torch.cat([latent_state.positions for latent_state in item["video_noisy_inputs"]], dim=0))
        ltx_ode_data_dict["video_clean_latents"].append(torch.cat([latent_state.clean_latent for latent_state in item["video_noisy_inputs"]], dim=0))
        
        ltx_ode_data_dict["audio_ode_latents"].append(torch.cat([latent_state.latent for latent_state in item["audio_noisy_inputs"]], dim=0))
        ltx_ode_data_dict["audio_denoise_mask"].append(torch.cat([latent_state.denoise_mask for latent_state in item["audio_noisy_inputs"]], dim=0))
        ltx_ode_data_dict["audio_positions"].append(torch.cat([latent_state.positions for latent_state in item["audio_noisy_inputs"]], dim=0))
        ltx_ode_data_dict["audio_clean_latents"].append(torch.cat([latent_state.clean_latent for latent_state in item["audio_noisy_inputs"]], dim=0))
        
        ltx_ode_data_dict["video_positive_context"].append(item["conditional_dict"]["v_context_p"])
        ltx_ode_data_dict["audio_positive_context"].append(item["conditional_dict"]["a_context_p"])
    
    for key in ltx_ode_data_dict.keys():
        ltx_ode_data_dict[key] = torch.stack(ltx_ode_data_dict[key], dim=0)
    
    return ltx_ode_data_dict


if __name__ == "__main__":
    dataset = ODERegressionStateDictDataset(["/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/ODE_sample_latents_3/"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, sampler=None, num_workers=0, collate_fn=ltx_collate_fn)
    dataloader = cycle(dataloader)
    local_dataloader_iterator = iter(dataloader)  # 重新创建迭代器
    import time
    start = time.time()
    while True:
        batch = next(local_dataloader_iterator)
        print(time.time() - start)
        import pdb;pdb.set_trace()
