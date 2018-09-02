import pandas as pd

from config import cfg


def get_hyperparametros_modelo(is_training):
    num_canais = get_num_canais()
    num_caracteristicas = get_num_caracteristicas()
    num_classes = get_num_classes()
    num_input = get_num_input(is_training)
    return num_canais, num_caracteristicas, num_classes, num_input

def maybe_donwload_and_extract():
    pass


def get_num_canais():
    return 1


def get_num_caracteristicas():
    return 28


def get_num_classes():
    if cfg.dataset == "fashionMNIST":
        return 10
    elif cfg.dataset == 'traffic_sign':
        return 62
    else:
        raise Exception('Dataset inválido, por favor confirme o nome do dataset:', cfg.dataset)


def get_num_input(is_training):
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


def get_caminho_resultado(is_training=True, is_cnn=True):
    if is_training:
        if is_cnn:
            return cfg.results + '/treinamentoCNN'
        else:
            return cfg.results + '/treinamentoCaps'
    else:
        if is_cnn:
            return cfg.results + '/avaliacaoCNN'
        else:
            return cfg.results + '/avaliacaoCaps'


def salvar_resultados(colunas, resultado, is_training, is_cnn):
    caminho_resultado = get_caminho_resultado(is_training, is_cnn) + "/resultado.csv"
    data_frame = pd.DataFrame(resultado, columns=colunas)
    data_frame.to_csv(caminho_resultado, index=False)
    return caminho_resultado
