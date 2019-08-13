
import pickle
import tensorflow as tf
import argparse
import os
import numpy as np

def make_model(i=11, l1=64, l2=2):
    inputs = tf.keras.Input(shape=(i,))
    x = tf.keras.layers.Dense(l1, activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Dense(l2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
    
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no-nesterov', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    print(args)
    
    model = make_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=args.lr, decay=args.decay, 
                                          momentum=args.momentum, 
                                          nesterov=not args.no_nesterov),
        loss='mean_squared_error',
        metrics=['mse'])
    print(model.summary())

    data = pickle.load(open(args.pkl_file, 'rb'))
    x_train = data['observations'][:int(0.8*len(data['observations']))]
    y_train = np.squeeze(data['actions'][:int(0.8*len(data['actions']))], axis=1)
    x_val = data['observations'][int(-0.2*len(data['observations'])):]
    y_val = np.squeeze(data['actions'][int(-0.2*len(data['actions'])):], axis=1)

    os.makedirs(args.ckpt, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.ckpt, 'weights.{epoch}-{val_loss:.4f}.hdf5'),
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