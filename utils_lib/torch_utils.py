import torch


def check_cuda():
    return torch.cuda.is_available()


CUDA_AVAILABLE = check_cuda()


def init_seeds(seed=0):
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available


def select_device(force_cpu=False, gpu_choice='0'):
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+ gpu_choice if CUDA_AVAILABLE else 'cpu')
        # set the GPU here @yangming
        torch.cuda.set_device(device)
    return device
