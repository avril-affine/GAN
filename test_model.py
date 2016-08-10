import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.clustering import KMeans


def get_feat_labels(sess, graph_def, option, model_ckpt, image_dir)

    if option == 'GAN':
        x_true_label = 'Discriminator/X_true:0'
        features_label = 'Discriminator/final_input:0'
    elif option == 'AE':
        x_true_label = 'input_image:0'
        features_label = 'features:0'
    else:
        raise Exception('Bad option inputted')

    x_true, feat_tensor = (
            tf.import_graph_def(graph_def,
                                return_elements=[x_true_label,
                                                 features_label],
                                name=''))
    image_data_tensor = tf.placeholder(tf.string)
    decode_tensor = tf.image.decode_jpeg(image_data_tensor,
                                         channels=INPUT_CHANNELS)

    label_names = [f for f in os.listdir(image_dir)
                   if not f.startswith('.')]
    features = []
    labels = []

    for label_i, label in enumerate(label_names):
        label_path = os.path.join(image_dir, label)
        image_files = [f for f in os.listdir(label_path)
                       if not f.startswith('.')]
        i = 0
        while i < len(image_files):
            batch_imgs = []
            batch_labels = []

            for _ in xrange(128):
                if i > len(image_files):
                    break
                image_path = os.path.join(label_path, image_files[i])
                image_data = gfile.FastGFile(image_path, 'rb').read()
                img = sess.run(decode_tensor,
                               feed_dict={image_data_tensor: image_data})
                img = img * 2. / 255. - 1.
                batch_imgs.append(img)
                batch_labels.append(label_i)
                i += 1

            batch_feats = sess.run(feat_tensor,
                                   feed_dict={x_true: batch_imgs})
            features.extend(batch_feats)
            labels.extend(batch_labels)
    
    return features, labels, label_names


def main():
    option = sys.argv[1]
    model_ckpt = sys.argv[2]
    image_dir = sys.argv[3]

    sess = tf.Session()

    with open(model_ckpt, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sess.graph.as_default()

    features, labels, label_names = get_feat_labels(sess,
                                                    graph_def,
                                                    option,
                                                    model_ckpt,
                                                    image_dir)

    inertias = []
    for n in xrange(2, 10):
        results = []
        for _ in xrange(5):
            km = KMeans(n)
            km.fit(features)
            results.append(km.intertia_)
        intertias.append(1. * sum(results) / len(results))

    plt.plot(range(2, 10), inertias)
    plt.xlabel('k clusters')
    plt.ylabel('clustering score')
    plt.title(option)

    plt.savefig(option + '_cluster_plot.png')
    plt.show()



if __name__ == '__main__':
    main()
