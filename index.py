import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
}
state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                      progress=True)


def alexnet_load_state_dict_trans(state_dict):
    features_map = {
        '0': 0,
        '3': 4,
        '6': 8,
        '8': 11,
        '10': 14,
    }
    alexnet_state_dict = {}
    for k in state_dict:
        module, idx, block = str(k).split('.')
        if module == 'features':
            alexnet_state_dict[f'features.{features_map[idx]}.{block}'] = state_dict[f'features.{idx}.{block}']
            alexnet_state_dict[f'features_.{features_map[idx]}.{block}'] = state_dict[f'features.{idx}.{block}']
    return alexnet_state_dict


def merge_state_dict(*archs):
    state_dict = {}
    for arch in archs:
        tmp_dict = load_state_dict_from_url(
            model_urls[arch], progress=True)
        if arch == 'alexnet':
            tmp_dict = alexnet_load_state_dict_trans(tmp_dict)
        state_dict.update(tmp_dict)
    return state_dict


print(merge_state_dict('alexnet', 'resnet18').keys())
