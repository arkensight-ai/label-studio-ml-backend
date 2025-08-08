# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import colorsys
import datetime
import functools
import io
import json
import os
import pickle
import subprocess
import time
import math
from collections import OrderedDict, defaultdict, deque
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor
import requests
import re
import tempfile
from urllib.parse import urlparse
import logging
from aule.misc import rgetattr

__torchvision_need_compat_flag = float(torchvision.__version__.split(".")[1]) < 7
if __torchvision_need_compat_flag:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
    
try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    from clearml.model import Model
except ImportError:
    Model = None
    
    
logger = logging.getLogger(__name__)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if os.environ.get("SHILONG_AMP", None) == "1":
            eps = 1e-4
        else:
            eps = 1e-6
        return self.total / (self.count + eps)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")

    return dist.group.WORLD


def all_gather_cpu(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [
        torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)
    ]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    if os.getenv("CPU_REDUCE") == "1":
        return all_gather_cpu(data)

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # print(name, str(meter))
            # import ipdb;ipdb.set_trace()
            if meter.count > 0:
                loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            # import ipdb; ipdb.set_trace()
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print_func(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    # import ipdb; ipdb.set_trace()
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(
            tensor.dim()
        )
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if (
        "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != ""
    ):  # 'RANK' in os.environ and
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = args.local_rank = int(os.environ["LOCAL_RANK"])

        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        # args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        # local_world_size = int(os.environ['GPU_PER_NODE_COUNT'])
        # args.world_size = args.world_size * local_world_size
        # args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        # args.rank = args.rank * local_world_size + args.local_rank
        print(
            "world size: {}, rank: {}, local rank: {}".format(
                args.world_size, args.rank, args.local_rank
            )
        )
        print(json.dumps(dict(os.environ), indent=2))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.local_rank = int(os.environ["SLURM_LOCALID"])
        args.world_size = int(os.environ["SLURM_NPROCS"])

        print(
            "world size: {}, world rank: {}, local rank: {}, device_count: {}".format(
                args.world_size, args.rank, args.local_rank, torch.cuda.device_count()
            )
        )
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print(
        "world_size:{} rank:{} local_rank:{}".format(
            args.world_size, args.rank, args.local_rank
        )
    )
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=args.rank,
        init_method=args.dist_url,
    )

    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def accuracy_onehot(pred, gt):
    """_summary_

    Args:
        pred (_type_): n, c
        gt (_type_): n, c
    """
    tp = ((pred - gt).abs().sum(-1) < 1e-4).float().sum()
    acc = tp / gt.shape[0] * 100
    return acc


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if __torchvision_need_compat_flag < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


class color_sys:
    def __init__(self, num_colors) -> None:
        self.num_colors = num_colors
        colors = []
        for i in np.arange(0.0, 360.0, 360.0 / num_colors):
            hue = i / 360.0
            lightness = (50 + np.random.rand() * 10) / 100.0
            saturation = (90 + np.random.rand() * 10) / 100.0
            colors.append(
                tuple(
                    [
                        int(j * 255)
                        for j in colorsys.hls_to_rgb(hue, lightness, saturation)
                    ]
                )
            )
        self.colors = colors

    def __call__(self, idx):
        return self.colors[idx]


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def clean_state_dict(state_dict, remove_query_generator_weight: bool = True):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'bert.embeddings.position_ids' in k:
            continue
        
        k = k.removeprefix("module.").removeprefix("model.")
        if remove_query_generator_weight and "dn_query_generator" in k:
            continue
        new_state_dict[k] = v
    return new_state_dict

def setup_state_dict(
    model: torch.nn.Module,
    state_dict: dict,
    ignore_parameters: tuple[str] = ("bert.embeddings.position_ids", "dn_query_generator"),
    peft_adapter_name = "default",
):
    new_state_dict = OrderedDict()
    all_model_param_names = set(model.state_dict().keys())
    peft_modules = set([k.rpartition("base_model.model.")[:2] for k in all_model_param_names if "base_model.model." in k])
    weight_starts_with_model = any([k.startswith("model.") for k in all_model_param_names])
    weight_starts_with_llm = any([k.startswith("llm.") for k in all_model_param_names])
    weight_starts_with_lmm = any([k.startswith("lmm.") for k in all_model_param_names])
    for k,v in state_dict.items():
        k = k.removeprefix("module.").replace("ori_model.", "base_layer.").replace("lora_A.", f"lora_A.{peft_adapter_name}.").replace("lora_B.", f"lora_B.{peft_adapter_name}.")
        if not weight_starts_with_model:
            k = k.removeprefix("model.")
        if not weight_starts_with_llm:
            k = k.removeprefix("llm.")
        if not weight_starts_with_lmm:
            k = k.removeprefix("lmm.")
        if peft_modules and "base_model.model." not in k:
            for peft_module in peft_modules:
                module_substring_to_replace = peft_module[0]
                k = k.replace(module_substring_to_replace, peft_module[0]+peft_module[1])
        for ignore in ignore_parameters:
            if ignore in k:
                continue
        if k in all_model_param_names:
            new_state_dict[k] = v
    return new_state_dict


def get_model_checkpoint(
    checkpoint_path: str,
    return_path: bool = False,
) -> dict:
    """
    Loads a model checkpoint from various sources and extracts the state dictionary.

    This function supports:
    1. Local file paths: Directly loads the file.
    2. Local directory paths: Looks for 'checkpoint/mp_rank_00_model_states.pt' within the directory.
    3. Google Cloud Storage (GS) links: Downloads the file from 'gs://' to a temporary location.
    4. HTTP/HTTPS links: Downloads the file from 'http(s)://' to a temporary location.
    5. ClearML Task IDs: Uses the ClearML SDK to get a local copy of the model.

    After obtaining a local file path, it checks if the loaded data contains
    a 'state_dict' or 'module' key and returns that sub-dictionary if found.
    Otherwise, it returns the entire loaded object.

    Args:
        checkpoint_path (str): The path or ID to the model checkpoint. Examples:
            - "my_model.pt"
            - "/path/to/my_model_dir"
            - "gs://my-bucket/models/model_v1.pt"
            - "https://example.com/checkpoints/best_model.pth"
            - "ef21d35ec3004407a1740453640bce3e" (ClearML Task ID)
        return_path (bool): If True, returns the local file path instead of the loaded data.

    Returns:
        dict: The loaded model state dictionary.

    Raises:
        ValueError: If the checkpoint path format is unrecognized or invalid.
        FileNotFoundError: If a local file or required sub-path does not exist.
        ImportError: If a required library (e.g., google-cloud-storage, clearml)
                     is not installed for the given checkpoint type.
        RuntimeError: For issues during download (GS, HTTP/HTTPS) or torch.load.
        requests.exceptions.RequestException: Specific errors during HTTP/HTTPS download.
        google.cloud.exceptions.GoogleCloudError: Specific errors during GS download.
    """
    final_local_path = None
    loaded_data = None
    is_temp_file = False

    if checkpoint_path.startswith("gs://"):
        if storage is None:
            raise ImportError(
                "The 'google-cloud-storage' library is required for GS links "
                "but is not installed. Please install it (e.g., pip install google-cloud-storage)."
            )
        try:
            parsed_gs = urlparse(checkpoint_path)
            bucket_name = parsed_gs.netloc
            blob_name = parsed_gs.path.lstrip('/')

            # Initialize Google Cloud Storage client
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download to a temporary file
            temp_file_obj = tempfile.NamedTemporaryFile(delete=False)
            blob.download_to_file(temp_file_obj)
            temp_file_obj.close() # Close the file handle before using its name
            final_local_path = temp_file_obj.name
            is_temp_file = True
        except Exception as e:
            raise RuntimeError(f"Failed to download from GS '{checkpoint_path}': {e}")

    elif checkpoint_path.startswith(("http://", "https://")):
        try:
            response = requests.get(checkpoint_path, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Download to a temporary file
            temp_file_obj = tempfile.NamedTemporaryFile(delete=False)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file_obj.write(chunk)
            temp_file_obj.close() # Close the file handle before using its name
            final_local_path = temp_file_obj.name
            is_temp_file = True
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download from HTTP(S) '{checkpoint_path}': {e}")

    # Check for ClearML ID (a 32-character hexadecimal string, typically a UUID)
    elif re.fullmatch(r"[0-9a-fA-F]{32}", checkpoint_path):
        if Model is None:
            raise ImportError(
                "The 'clearml' library is required for ClearML IDs but is not installed. "
                "Please install it (e.g., pip install clearml)."
            )
        try:
            clearml_model = Model(checkpoint_path)
            final_local_path = clearml_model.get_local_copy()
            if not final_local_path:
                raise ValueError(f"ClearML Model.get_local_copy() returned no path for ID: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to get checkpoint from ClearML ID '{checkpoint_path}': {e}")

    else:
        final_local_path = checkpoint_path

    if not final_local_path:
        raise ValueError("Could not determine a valid local path for the provided checkpoint_path.")

    if os.path.isdir(final_local_path):
        checkpoint_subdir = os.path.join(final_local_path, "checkpoint")
        target_file = os.path.join(checkpoint_subdir, "mp_rank_00_model_states.pt")

        if os.path.isdir(checkpoint_subdir) and os.path.exists(target_file):
            final_local_path = target_file
        else:
            raise FileNotFoundError(
                f"The directory '{final_local_path}' was provided, but it does not "
                f"contain the expected 'checkpoint/mp_rank_00_model_states.pt' file."
            )
    elif not os.path.exists(final_local_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {final_local_path}")
    
    if return_path:
        return final_local_path

    try:
        loaded_data = torch.load(final_local_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{final_local_path}' with torch.load: {e}")
    finally:
        if is_temp_file and os.path.exists(final_local_path):
            try:
                os.remove(final_local_path)
            except OSError as e:
                pass

    if isinstance(loaded_data, dict):
        if 'state_dict' in loaded_data:
            return loaded_data['state_dict']
        elif 'module' in loaded_data:
            return loaded_data['module']
    
    return loaded_data

def load_checkpoint_by_module(
    model: torch.nn.Module,
    checkpoint_loading_config: dict[str, str],
    strict: bool = True,
) -> None:
    for key, value in checkpoint_loading_config.items():
        if not value:
            continue
        try:
            module = rgetattr(model, key)
        except AttributeError as e:
            logger.error(f"IGNORING CHECKPOINT LOADING!!! Failed to get module '{key}' from model: {e} when trying to load the checkpoint")
            continue
        pytorch_checkpoint = get_model_checkpoint(value)
        log = module.load_state_dict(
            setup_state_dict(module, pytorch_checkpoint), strict=strict
        )
        logger.info(f"Loaded checkpoint from {value} on module: {key}, log: {log}")


def split_by_mask(tensor, mask):
    # Find the indices where splits should occur (where mask is False)
    split_indices = (mask).nonzero().squeeze(-1).tolist()

    # Split the tensor along the last dimension
    result = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        # Only include segments that correspond to True in the mask
        if end > start + 1:
            result.append(tensor[..., start + 1 : end].mean())

    return torch.tensor(result)


def coordinate_to_encoding(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    dim_t = torch.arange(
        num_feats, dtype=coord_tensor.dtype, device=coord_tensor.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos
