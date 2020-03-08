import datetime, os
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from PIL import Image

config = Config()

def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

def read_images(dataset_path, mode):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        with open(dataset_path) as f:
            data = f.read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg') or sample.endswith('.bmp') or sample.endswith('.png'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append([label, label, label, label, label])
            label += 1
    else:
        raise Exception("Unknown mode.")

    images = list()
    for p in imagepaths:
        image = Image.open(p)
        if image is None:
            continue
        image = image.convert('RGB')
        image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
        image = np.asarray(image)
        image = np.float32(image)/127.5 - 1
        #image = np.expand_dims(image, axis=2)
        images.append(image)

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    print("images: %d" % len(images))
    print("labels: %d" % len(labels))

    if len(images) != len(labels):
        print("Error---------------------")

    else:
        return images, labels

def data_shuffle(images, labels):
    s = np.arange(images.shape[0])
    np.random.shuffle(s)
    return images[s], labels[s]

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def save_plot_generated(sample_data, n, str):
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow((sample_data[i]).reshape([config.IMG_SIZE, config.IMG_SIZE, 3]))
    plt.savefig(str, bbox_inches='tight', pad_inches=0)
    plt.close()
