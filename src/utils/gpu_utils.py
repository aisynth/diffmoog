import torch


def get_device(gpu_index: int = 0):
    if int(gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')
    return device


def move_to(obj, device):
    if torch.is_tensor(obj):
        obj = obj.to(device)
        return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = move_to(v, device)
        return obj
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            obj[idx] = move_to(v, device)
        return obj
    else:
        return obj
