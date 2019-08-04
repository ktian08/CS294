
import pickle
import tensorflow as tf
import argparse
import os
import numpy as np
tf.enable_eager_execution()

class Model(tf.keras.Model):
    def __init__(self, hidden_units):
        super(Model, self).__init__()
        self.l1 = tf.keras.layers.Dense(hidden_units, input_shape=(111,), activation=tf.nn.relu)
        self.l2 = tf.keras.layers.Dense(8)
        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
    
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden-units', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no-nesterov', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    
    model = Model(args.hidden_units)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=args.lr, decay=args.decay, 
                                          momentum=args.momentum, 
                                          nesterov=not args.no_nesterov),
        loss='mean_squared_error',
        metrics=['mse'])

    data = pickle.load(open(args.pkl_file, 'rb'))
    x_train = np.expand_dims(data['observations'][int(0.8*len(data['observations'])):], axis=1)
    y_train = data['actions'][int(0.8*len(data['actions'])):]
    x_val = np.expand_dims(data['observations'][int(-0.2*len(data['observations'])):], axis=1)
    y_val = data['actions'][int(-0.2*len(data['actions'])):]
    
    os.makedirs(args.ckpt, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        args.ckpt,
        save_weights_only=True,
        verbose=1)

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[cp_callback])
    return history
        
if __name__=='__main__':
    train()