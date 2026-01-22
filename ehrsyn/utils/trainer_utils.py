import logging, os, json
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from dataclasses import dataclass
from ehrsyn.utils.utils import extract_sep_idx_from_event

logger = logging.getLogger(__name__)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, 
                patience=7, 
                verbose=True, 
                delta=1e-4, 
                compare='increase',
                metric='auprc'

                ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metric_min = 0
        self.delta = delta
        self.compare_score = self.increase if compare=='increase' else self.decrease
        self.metric = metric

    def __call__(self, target_metric):
        update_token=False
        score = target_metric

        if self.best_score is None:
            self.best_score = score

        if self.compare_score(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation {self.metric} {self.compare_score.__name__}d {self.target_metric_min:.6f} --> {target_metric:.6f})')
            self.target_metric_min = target_metric
            self.counter = 0
            update_token = True
        
        return update_token

    def increase(self, score):
        if score < self.best_score*(1+self.delta):
           return True
        else:
           return False

    def decrease(self, score):
        if score > self.best_score*(1+self.delta):
            return True
        else:
           return False


def count_parameters(model):
    """
    Count the total number of parameters and the number of trainable parameters in a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def save_model(path, model, optimizer, epoch):
    """
    Save the model and optimizer states to a file.

    Args:
        path (str): The base path for saving the model.
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        epoch (int): The current epoch number.
    """
    # Adjust path to include file extension
    save_path = f'{path}.pkl'

    # Check if model is wrapped in DataParallel or DistributedDataParallel
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Save the model and optimizer state dictionaries
    torch.save({
        'model_state_dict': model_state_dict,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

    print(f'Model saved at: {save_path}')

def load_model(path, model, optimizer=None):
    """
    Load a model and optimizer states from a file.

    Args:
        path (str): The path to the saved model file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.

    Returns:
        tuple: Returns the epoch number, model, and optimizer.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Append file extension
    full_path = f'{path}.pkl'

    # Check if the file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No checkpoint file found at {full_path}")

    # Load the checkpoint
    checkpoint = torch.load(full_path, map_location='cpu')
    epoch = checkpoint['epoch'] + 1

    # Prepare the state dictionary
    state_dict = checkpoint['model_state_dict']
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    # Load state into the model
    model.load_state_dict(state_dict)

    # Load state into the optimizer if provided
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {full_path}")
    return epoch, model, optimizer
    
def save_config(config, path):
    with open(f'{path}.json', 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Config saved at: {path}.json')

def load_config(path):
    with open(f'{path}.json', 'r') as f:
        config = json.load(f)
    print(f'Config loaded from: {path}.json')
    return config

def log_from_dict(epoch_log, data_type, epoch):
    summary = {'epoch':epoch}
    for key, values in epoch_log.items():
        
        if isinstance(values, dict):
            for k, v in values.items():
                summary[f'{data_type}/{key}_{k}'] = v    
                print(f'{epoch}:\t{data_type}/{key}_{k} : {v:.3f}')
        else:
            summary[f'{data_type}/{key}'] = values
            print(f'{epoch}:\t{data_type}/{key} : {values:.3f}')
        
    return summary


class ExperimentCheckpoint:
    def __init__(self, config):
        self.config = config

    def generate_checkpoint_path(self):
        ckpt_name, _ = self._generate_checkpoint_name()
        return ckpt_name, os.path.join(self.config['output_path'], ckpt_name)

    def _generate_checkpoint_name(self):
        exp_component = self._generate_exp_component()
        return '_'.join(map(str, exp_component)), exp_component

    def _generate_exp_component(self):
        required_keys = ['exp_name']
        exp_component = [self.config[key] for key in required_keys]
        return exp_component

    

class SaveNumpy:
    def __init__(self, config, ckpt_name):
        self.config = config
        self.folder_name = ckpt_name
        self.input_path = config["real_input_path"]
        self.max_event_size = config["max_event_size"]
        self.max_event_token_len = config["max_event_token_len"]
        self.spatial_dim = config["spatial_dim"]

        self._prepare_filename()

        self.targets = config["save_as_numpy"].split(",") if config["save_as_numpy"] else []
        self.output = {target: [] for target in self.targets}

        if self.config["require_gt_time"]:
            assert "time_logits" not in self.targets, "time_logits should not be included when require_gt_time is True"
            self.output["time_logits"] = []

    def _prepare_filename(self):
        if not self.config["save_as_numpy"]:
            return
        if self.config['sample']:
            self.folder_name += f"_top{self.config['topk']}_{self.config['temperature']}"

    def concat(self, net_output, targets):
        for target in self.targets:    
            if target in ["input_logits", "type_logits", "dpe_logits"]:
                pred = torch.argmax(net_output[target], dim=-1).detach().cpu().numpy().reshape(-1, self.max_event_size, self.max_event_token_len)

            elif target in ["enc_indices"]:
                pred = net_output[target].detach().cpu().numpy().reshape(-1, self.max_event_size, self.spatial_dim*self.config["num_quantizers"])

            elif target in ["time_logits"]:
                time_pred = torch.argmax(net_output[target], dim=-1).detach().cpu().numpy()
                time_pred = time_pred.reshape(-1, self.config["max_event_size"], self.config["time_len"])

                num_codebooks_offset = self.config["num_codebooks"] + 3
                time_pred -= num_codebooks_offset
                
                # 保持2维或3维结构，不合并成标量（与真实数据格式一致）
                # 真实数据格式：每个样本是 (max_event_size, time_len) 的数组
                pred = time_pred.copy()  # 保持 (batch, max_event_size, time_len) 结构
                
                # Mask for padding values
                pad_mask = pred < 0
                pred[pad_mask] = -100
                
            self.output[target].append(pred)

        if self.config["require_gt_time"]:
            self.output["time_logits"].append(targets["time_ids"].detach().cpu().numpy())

    
    def clean_output(self):
        """
        Cleans the output data by ensuring that pad events and tokens are correctly marked.
        This is done at both the event and token levels.
        """
        # 优化：如果只有 enc_indices，跳过所有 input/type/dpe 相关的处理
        only_enc_indices = len(self.targets) == 1 and "enc_indices" in self.targets
        
        if only_enc_indices:
            # 只处理 enc_indices，直接 vstack 即可
            logger.info("只保存 enc_indices，跳过 input/type/dpe 处理")
            self.output["enc_indices"] = np.vstack(self.output["enc_indices"])
            return
        
        PAD_EVENT_ID = -100  # Identifier for padding events (code-only case)
        id2word_path = f"{self.config['ehr']}_id2word.pkl"
        id2word = pd.read_pickle(os.path.join(self.input_path, id2word_path))
        
        # Stack target outputs vertically
        output_keys = self.config['save_as_numpy'].split(",") 
        

        for target in self.targets: 
            if target == "time_logits" and self.config.get("time_len", 2) in [2, 3]:
                # 对于时间数据，保持2维或3维结构，转换为object数组（与真实数据格式一致）
                # 将 (N, max_event_size, time_len) 转换为 object数组，每个元素是 (max_event_size, time_len)
                time_data = np.vstack(self.output[target])  # (total_samples, max_event_size, time_len)
                # 转换为object数组，每个样本是一个2维数组
                time_list = [time_data[i] for i in range(time_data.shape[0])]
                self.output[target] = np.array(time_list, dtype=object)
            else:
                self.output[target] = np.vstack(self.output[target])

        if self.config["require_gt_time"]:
            if self.config.get("time_len", 2) in [2, 3]:
                # 保持2维或3维结构
                time_data = np.vstack(self.output["time_logits"])
                time_list = [time_data[i] for i in range(time_data.shape[0])]
                self.output["time_logits"] = np.array(time_list, dtype=object)
            else:
                self.output["time_logits"] = np.vstack(self.output["time_logits"])
            output_keys.append("time_logits")

        iterable = (tqdm(self.output[key]) if i == 0 else self.output[key] for i, key in enumerate(output_keys))
        for items in zip(*iterable):
            items_dict = {key: item for key, item in zip(output_keys, items)}

            if 'time_logits' in items_dict:
                # 处理2维或3维时间数据
                if isinstance(items_dict['time_logits'], np.ndarray) and items_dict['time_logits'].dtype == object:
                    # object数组格式：每个元素是一个2维数组
                    time_array = items_dict['time_logits']
                    # 找到第一个全为-100的行（padding）
                    pad_event_start_idx = len(time_array)
                    for i, time_row in enumerate(time_array):
                        if np.all(time_row == -100):
                            pad_event_start_idx = i
                            break
                else:
                    # 1维数组格式（旧格式）
                    pad_event_indices = np.where(items_dict['time_logits'] == PAD_EVENT_ID)[0]
                    if pad_event_indices.size > 0:
                        pad_event_start_idx = pad_event_indices[0]
                    else:
                        pad_event_start_idx = len(items_dict['time_logits'])
                
                for key in output_keys:
                    if key == "enc_indices":
                        items_dict[key][pad_event_start_idx:] = self.config["num_codebooks"]
                    elif key in ["input_logits", "type_logits", "dpe_logits"]:
                        items_dict[key][pad_event_start_idx:] = 0
            else:
                pad_event_start_idx = len(items_dict["input_logits"])

            for i in range(pad_event_start_idx):
                items_dict['input_logits'][i] = np.vectorize(id2word.get)(items_dict['input_logits'][i])
                sep_idx = extract_sep_idx_from_event(items_dict['input_logits'][i], items_dict['type_logits'][i])
                if sep_idx is not None:
                    for key in ['input_logits', 'type_logits', 'dpe_logits']:
                        items_dict[key][i][sep_idx+1:] = 0

    def save(self):
        if self.config["save_as_numpy"]:
            self.clean_output()
            target_filename = {
                "input_logits": f"{self.config['ehr']}_{self.config['structure']}_input.npy",
                "type_logits": f"{self.config['ehr']}_{self.config['structure']}_type.npy",
                "dpe_logits": f"{self.config['ehr']}_{self.config['structure']}_dpe.npy",
                "enc_indices": f"{self.config['ehr']}_{self.config['structure']}_code.npy",
                "time_logits": f"{self.config['ehr']}_{self.config['structure']}_time.npy",
            }

            for target in self.targets:
                output_file = target_filename[target]
                output_dir = os.path.join(self.config['generated_data_path'], self.folder_name)
                output_path = os.path.join(output_dir, output_file)
            
                os.makedirs(output_dir, exist_ok=True)
                np.save(output_path, self.output[target])
                logger.info(f"Saved {self.output[target].shape} samples to {output_path}")
        
            return output_dir


class StreamingSaveNumpy:
    def __init__(self, config, ckpt_name, dataloader=None):
        self.config = config
        self.output_dir = os.path.join(
            config['generated_data_path'], 
            ckpt_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 使用memmap进行流式写入
        total_samples = self._get_total_samples(dataloader)
        self.enc_indices_file = os.path.join(
            self.output_dir, 
            f"{config['ehr']}_{config['structure']}_code.npy"
        )
        
        # 预分配空间
        shape = (
            total_samples, 
            config['max_event_size'], 
            config['spatial_dim'] * config['num_quantizers']
        )
        self.enc_indices_mmap = np.lib.format.open_memmap(
            self.enc_indices_file,
            mode='w+',
            dtype=np.int16,
            shape=shape
        )
        self.current_idx = 0
    
    def _get_total_samples(self, dataloader=None):
        """获取总样本数"""
        if dataloader is not None:
            # 如果提供了数据加载器，从数据加载器中获取
            return len(dataloader.dataset)
        elif 'total_samples' in self.config:
            # 如果配置中有总样本数，直接使用
            return self.config['total_samples']
        else:
            # 尝试从输入文件中读取
            try:
                input_path = self.config.get('real_input_path', self.config.get('input_path'))
                code_file = os.path.join(
                    input_path,
                    f"{self.config['ehr']}_{self.config['structure']}_code.npy"
                )
                if os.path.exists(code_file):
                    code_data = np.load(code_file, allow_pickle=True)
                    return len(code_data)
            except Exception as e:
                logger.warning(f"无法从输入文件获取总样本数: {e}")
            
            # 如果都失败了，返回一个默认值（需要后续调整）
            logger.warning("无法确定总样本数，使用默认值1000000，请确保实际样本数不超过此值")
            return 1000000
    
    def append(self, enc_indices):
        """流式追加数据"""
        # 处理torch tensor或numpy数组
        if torch.is_tensor(enc_indices):
            enc_indices_np = enc_indices.detach().cpu().numpy()
        else:
            enc_indices_np = np.asarray(enc_indices)
        
        batch_size = enc_indices_np.shape[0]
        end_idx = self.current_idx + batch_size
        
        # 确保不超过预分配的空间
        if end_idx > self.enc_indices_mmap.shape[0]:
            logger.warning(f"样本数超过预分配空间 ({self.enc_indices_mmap.shape[0]})，截断到 {self.current_idx}")
            end_idx = self.enc_indices_mmap.shape[0]
            batch_size = end_idx - self.current_idx
            if batch_size <= 0:
                return
        
        self.enc_indices_mmap[self.current_idx:end_idx] = \
            enc_indices_np.reshape(
                batch_size, 
                self.config['max_event_size'], 
                -1
            )
        
        self.current_idx = end_idx
        
        # 定期flush
        if self.current_idx % 1000 == 0:
            self.enc_indices_mmap.flush()
    
    def finalize(self):
        """完成写入"""
        self.enc_indices_mmap.flush()
        # 如果实际样本数小于预分配的空间，需要截断文件
        if self.current_idx < self.enc_indices_mmap.shape[0]:
            logger.info(f"实际样本数 ({self.current_idx}) 小于预分配空间 ({self.enc_indices_mmap.shape[0]})，截断文件...")
            # 使用memmap读取并保存截断后的数据
            temp_file = self.enc_indices_file + '.tmp'
            truncated_mmap = np.lib.format.open_memmap(
                temp_file,
                mode='w+',
                dtype=self.enc_indices_mmap.dtype,
                shape=(self.current_idx,) + self.enc_indices_mmap.shape[1:]
            )
            truncated_mmap[:] = self.enc_indices_mmap[:self.current_idx]
            truncated_mmap.flush()
            del truncated_mmap
            del self.enc_indices_mmap
            # 替换原文件
            import shutil
            shutil.move(temp_file, self.enc_indices_file)
            logger.info(f"文件已截断为 ({self.current_idx}, {self.config['max_event_size']}, {self.config['spatial_dim'] * self.config['num_quantizers']})")
        else:
            logger.info(f"Saved {self.current_idx} samples to {self.enc_indices_file}")
         