import torch


def load_pretrain(net, pth_path):
    '''
    weird here
    :param net:
    :param pth_path:
    :return:
    '''
    checkpoint = torch.load(pth_path)

    net_dict = net.state_dict()

    pretrained_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'fc' in k:
            continue
        pretrained_dict.setdefault(k[7:], checkpoint['state_dict'][k])

    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)