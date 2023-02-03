import tensorflow as tf
import argparse


from scripts.utils.Generators import realsense_generator
from scripts.utils.losses import MSE_loss

from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet import preprocess_input



def depth_encoder_model(depth_in_shape, num_unfreeze=0):
    depth_in = layers.Input(depth_in_shape)

    inner_in = preprocess_input(depth_in)
    FE = MobileNetV2(input_shape=depth_in_shape, include_top=False, weights='imagenet')

    for layer in FE.layers:
        layer.trainable = False

    for layer_num in range(num_unfreeze):
        layer_num += 1
        FE.layers[-layer_num].trainable = True

    featmap = FE(inner_in)

    model = tf.keras.Model(inner_in, featmap, name='depth_encode')
    model.summary()
    return model


def decoder_model(latent_shape):
    decoder_in = layers.Input(latent_shape)

    depth_restored = layers.Conv2DTranspose(512, 2, 2)(decoder_in)
    depth_restored = layers.LeakyReLU()(depth_restored)

    depth_restored = layers.Conv2DTranspose(256, 2, 2)(depth_restored)
    depth_restored = layers.LeakyReLU()(depth_restored)

    depth_restored = layers.Conv2DTranspose(128, 2, 2)(depth_restored)
    depth_restored = layers.LeakyReLU()(depth_restored)

    depth_restored = layers.Conv2DTranspose(64, 2, 2)(depth_restored)
    depth_restored = layers.LeakyReLU()(depth_restored)

    depth_restored = layers.Conv2DTranspose(32, 2, 2)(depth_restored)
    depth_restored = layers.LeakyReLU()(depth_restored)

    depth_restored = layers.Conv2D(3, 1)(depth_restored)
    depth_restored = layers.LeakyReLU()(depth_restored)

    model = tf.keras.Model(decoder_in, depth_restored, name='depth_restore')
    model.summary()
    return model



def trainD2D(args):
    '''
    Trains a depth image to depth image autoencoder
    :param args: various runtime args
    :return: trained auto encoder model
    '''
    train_path = args.train_image_path
    model_path = args.model_path
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    img_size = args.img_size

    depth_in = layers.Input(img_size)

    depth_encoder = depth_encoder_model(depth_in.shape[1:])

    decoder = decoder_model(depth_encoder.output_shape[1:])

    latent = depth_encoder(depth_in)
    depth = decoder(latent)

    model = tf.keras.Model(depth_in, depth, name='RGB2DEPTH')

    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, epsilon=0.01)

    model.compile(optimizer, MSE_loss, run_eagerly=False)
    model.summary()

    train_gen = realsense_generator(3, train_path, img_size, shuffle=True, augment=False)
    tensorboard_callback = tf.keras.callbacks.TensorBoard('logs/depth2depth')
    model_save = tf.keras.callbacks.ModelCheckpoint(model_path, 'loss')
    callbacks = [tensorboard_callback, model_save]

    model.fit(train_gen, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_image_path', type=str, help='path to test image')
    parser.add_argument('model_path', type=str, help='path to save trained models')
    parser.add_argument('--batch_size', type=int, help='batch size when training autoencoder', default=1)
    parser.add_argument('--lr', type=float, help='learning rate when training autoencoder', default=0.001)
    parser.add_argument('--epochs', type=int, help='epochs when training autoencoder', default=99)
    parser.add_argument('--img_size', type=tuple, help='size and number of channels of input image',
                        default=(256, 256, 3))

    args = parser.parse_args()

    trainD2D(args)

