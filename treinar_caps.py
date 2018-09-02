import tensorflow as tf

import utils
from config import cfg
from input import get_batch_data
from modeloCapsulas import CapsNet


def train():
    iterator = get_batch_data(cfg.dataset, cfg.batch_size, is_training=True)

    imagem, label = iterator.get_next()

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_hyperparametros_modelo(is_training=True)

    capsNet = CapsNet(num_caracteristicas, num_caracteristicas, num_canais, num_classes)

    capsNet.create_network(imagem, label)

    loss, train_ops, summary_ops = capsNet.train()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(utils.get_caminho_resultado(True, False), sess.graph)
        total_batch = num_input // cfg.batch_size
        colunas = ["Epoch", "Custo", "Precisao"]
        resultado = []
        for i in range(cfg.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for batch in range(total_batch):
                _, custo, acc, summary, step = sess.run([train_ops, loss, capsNet.accuracy, summary_ops,
                                                         capsNet.global_step])
                summary_writer.add_summary(summary, step)
                avg_cost += custo / total_batch
                avg_acc += acc / total_batch
            resultado.append([i + 1, avg_cost, avg_acc])
            print("Epoch " + str(i) + ", Custo= {:.6f}".format(avg_cost) + ", Precisao do treinamento= {:.5f}".format(
                avg_acc))
        caminho = utils.salvar_resultados(colunas, resultado, True, False)
        print("CSV salvo em {}".format(caminho))
        save_path = saver.save(sess, cfg.results + "/model.ckpt")
        print("Modelo salvo em: %s" % save_path)



def main(argv=None):
    # cifar10.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == "__main__":
    tf.app.run()
