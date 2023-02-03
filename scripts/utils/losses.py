import tensorflow as tf

# tf.config.run_functions_eagerly(True)

def MSE_loss(y_true, y_pred):
    MSE_loss_func = tf.keras.losses.MeanSquaredError(reduction='none')
    MSE_loss = MSE_loss_func(y_true, y_pred)
    return MSE_loss