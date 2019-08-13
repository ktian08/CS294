#!/usr/bin/env python

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import bcloning

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('model_weights', type=str)
    parser.add_argument('train_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session() as sess:
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        
        model = bcloning.make_model(i=111, l1=64, l2=8)
        model.load_weights(args.model_weights)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, 
                                          momentum=0.9,
                                          nesterov=True),
            loss='mean_squared_error',
            metrics=['mse'])
        
        expert_data = pickle.load(open(args.train_file, 'rb'))
        expert_data['observations'] = np.expand_dims(expert_data['observations'], axis=1)

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = np.expand_dims(obs, axis=0)
                observations.append(obs)
                action = model.predict(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

            expert_data['observations'] = np.append(expert_data['observations'], 
                                                    np.array(observations), 
                                                    axis=0)
            expert_data['actions'] = np.append(expert_data['actions'], 
                                               np.array(actions), 
                                               axis=0)
            bcloning.train_dagger(model, expert_data, epochs=1, batch=1)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
if __name__ == '__main__':
    main()
