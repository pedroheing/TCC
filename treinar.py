import tensorflow as tf

import utils
from config import cfg
from input import get_batch_data
from modelo import ConvolutionalNeuralNetwork


def train():
    global_step = tf.train.get_or_create_global_step()

    imagens, labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads, is_training=True)

    cnn = ConvolutionalNeuralNetwork()

    logits = cnn.construir_arquitetura(imagens)

    custo = cnn.custo(logits, labels)

    treinamento = cnn.treinar(custo, global_step)

    accuracy = cnn.precisao(logits, labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        summary_writer = tf.summary.FileWriter(cfg.results + '/treinamento', sess.graph)
        total_batch = utils.get_tamanho_dataset(is_training=True) // cfg.batch_size
        menor_erro = 1.
        for i in range(cfg.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for batch in range(total_batch):
                merge = tf.summary.merge_all()
                sumario, _, step = sess.run([merge, treinamento, global_step])
                summary_writer.add_summary(sumario, step)
                loss, acc = sess.run([custo, accuracy])
                avg_cost += loss / total_batch
                avg_acc += acc / total_batch
            print("Epoch " + str(i) + ", Custo= " + \
                  "{:.6f}".format(avg_cost) + ", Precisao do treinamento= " + \
                  "{:.5f}".format(avg_acc))
            if avg_cost < menor_erro:
                menor_erro = avg_cost
                save_path = saver.save(sess, cfg.results + "/model.ckpt")
        print("Modelo salvo em: %s" % save_path)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    # cifar10.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == "__main__":
    tf.app.run()
