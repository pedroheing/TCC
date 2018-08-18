from config import cfg


def maybe_donwload_and_extract():
    pass


def get_tamanho_dataset(is_training):
    if cfg.dataset == "fashionMNIST":
        if is_training:
            return 55000
        else:
            return 10000
    elif cfg.dataset == 'traffic_sign':
        if is_training:
            return 4575
        else:
            return 2520
    else:
        raise Exception('Dataset inválido, por favor confirme o nome do dataset:', cfg.dataset)
