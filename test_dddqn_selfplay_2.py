from environments.snake_env import SnakeEnvironment
from dddqn_model.dddqn import *
import numpy as np
import os
import tensorflow as tf


saving_path = "models_selfplay_2"  # The path to save our PPO_implementation to.
load_model_path = 'models_selfplay_2'

batch_size = 512 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 0.1 # Starting chance of random action
endE = 0.0001 # Final chance of random action
annealing_steps = 500000. # How many steps of training to reduce startE to endE.
num_episodes = 200 # How many episodes of game environment to train network with.
pre_train_steps = 5000 # How many steps of random actions before training begins.
max_epLength = 1000000 # The max amount of steps in an episode.
h_size = 1296*2 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network


def main():

    # Environment setting
    spacing = 22
    dimensions = 15
    history = 4
    env = SnakeEnvironment(num_agents=2, num_fruits=3, spacing=spacing, dimensions=dimensions, flatten_states=False,
                     reward_killed=-1.0, history=history)
    env.reset()

    # Tensorflow PPO_implementation setting
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Create Main Agent
    agent_1 = Qnetwork(h_size=h_size, scope="main_agent_1")

    # Create Adversary Agent
    agent_2 = Qnetwork(h_size=h_size, scope="main_agent_2")

    # Trainable variable for pretrain agent, agent1, agent2
    trainables = tf.compat.v1.trainable_variables()

    weights_agent_1 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_1', 'target_agent_1']]

    weights_agent_2 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_2']]

    # Copy weight of agent1 to agent2
    update_weights = [tf.compat.v1.assign(weights_2, weights_1) for (weights_2, weights_1) in zip(weights_agent_2, weights_agent_1)]

    # Tensorflow Saver for agent2
    saver_new_model = tf.compat.v1.train.Saver(weights_agent_1)

    # Create lists to contain total rewards and steps per episode
    reward_list_agent_2 = []
    reward_list_agent_1 = []

    # Make a path for our PPO_implementation to be saved in.
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.compat.v1.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(load_model_path)
        saver_new_model.restore(sess, ckpt.model_checkpoint_path)
        sess.run(update_weights)
        print('Loading Saving Model...')
        print(ckpt.model_checkpoint_path)


        for i in range(num_episodes):

            # Matching agent1, agent2 for Self-Play
            for k in range(0, 100):

                # Reset environment and get first new observation
                obs = env.reset()

                # Initialize state, reward, end flag of agent1, agent2
                obs_agent_1 = obs[1]
                obs_agent_2 = obs[0]


                # Initialize sum of reward of agent1, agent2
                sum_rewards_agent_2 = 0
                sum_rewards_agent_1 = 0

                # The Q-Network
                for j in range(0, max_epLength):
                    env.render()

                    # Select action of agent 2
                    action_agent_2 = sess.run(agent_2.predict, feed_dict={agent_2.imageIn: [obs_agent_2 / 3.0]})[
                        0]

                    # Select action of agent 1
                    action_agent_1 = sess.run(agent_1.predict, feed_dict={agent_1.imageIn: [obs_agent_1 / 3.0]})[
                        0]

                    # Move agent1, agent2 and get reward, next state, end flag of each agent and end flag of current episode
                    next_state, reward, done, d_common = env.step([action_agent_2, action_agent_1])


                    # Reward of agent1, agent2
                    reward_agent_2 = reward[0]
                    reward_agent_1 = reward[1]

                    # End flag of agent1, agent2
                    d_agent_2 = done[0]
                    d_agent_1 = done[1]


                    # Next state of agent1, agent2
                    next_state_agent_2 = next_state[0]
                    next_state_agent_1 = next_state[1]

                    # Add a reward of each agent to total reward
                    sum_rewards_agent_2 += reward_agent_2
                    sum_rewards_agent_1 += reward_agent_1

                    # Save a next state to current state for next step
                    obs_agent_2 = next_state_agent_2
                    obs_agent_1 = next_state_agent_1


                    # End episode if both agents are dead
                    if (d_agent_2 == True and d_agent_1 == True):
                        break

                # Save sum of each agent for printing performance
                reward_list_agent_2.append(sum_rewards_agent_2)
                reward_list_agent_1.append(sum_rewards_agent_1)



            # Periodically print performance of agents
            if len(reward_list_agent_1) % 50 == 0:
                print(i, "agent_2", np.mean(reward_list_agent_2[-10:]))
                print(i, "agent_1", np.mean(reward_list_agent_1[-10:]))
                print("")


if __name__ == '__main__':

    main()