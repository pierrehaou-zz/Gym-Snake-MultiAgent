from environments.snake_env import SnakeEnvironment
import os
from dddqn_model.dddqn import *

batch_size = 512 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 0.1 # Starting chance of random action
endE = 0.0001 # Final chance of random action
annealing_steps = 500000. # How many steps of training to reduce startE to endE.
num_episodes = 20000 # How many episodes of game environment to train network with.
pre_train_steps = 50000 # How many steps of random actions before training begins.
max_epLength = 1000000 # The max amount of steps allowed in the episode.
load_model = False # Whether to load a saved PPO_implementation.
load_model_path = 'models_single'
saving_path = 'models_single'  # The path to save our PPO_implementation to.
h_size = 1296*2 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network

def main():

    print("Training has begun!")

    spacing = 22
    dimensions = 15
    history = 4
    env = SnakeEnvironment(num_agents=1, num_fruits=3, spacing=spacing, dimensions=dimensions, flatten_states=False,
                     reward_killed=-1.0, history=history)
    env.reset()

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    mainQN = Qnetwork(h_size=h_size, scope="main_new")
    targetQN = Qnetwork(h_size=h_size, scope="target_new")

    init = tf.compat.v1.global_variables_initializer()
    saver_new_model = tf.compat.v1.train.Saver()
    trainables = tf.compat.v1.trainable_variables()
    variables_new_restore = [v for v in trainables if v.name.split('/')[0] in ['main_new', 'target_new']]
    targetOps = updateTargetGraph(variables_new_restore, tau)
    myBuffer = experience_buffer()

    # Set the rate of random action decrease.
    explore = startE
    stepDrop = (startE - endE) / annealing_steps

    # create lists to contain total rewards
    reward_list = []
    total_steps = 0

    # Make a path for our PPO_implementation to be saved in.
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    current_best = 50  # If PPO_implementation outperforms this number, that PPO_implementation will be saved

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(load_model_path)
            saver_new_model.restore(sess, ckpt.model_checkpoint_path)
            print(f'loaded PPO_implementation at {ckpt.model_checkpoint_path}')

        for i in range(num_episodes):
            episodeBuffer = experience_buffer()

            # Reset environment and get first new observation
            observation = env.reset()
            reward_all = 0
            current_step = 0

            # The Q-Network
            while current_step < max_epLength:

                current_step += 1

                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < explore or total_steps < pre_train_steps:
                    action = np.random.randint(0, 4)
                else:
                    action = sess.run(mainQN.predict, feed_dict={mainQN.imageIn: [observation / 3.0]})[0]

                next_observation, reward, _, done = env.step(action)
                total_steps += 1

                # Save the experience to our episode buffer.
                episodeBuffer.add(np.reshape(np.array([observation, action, reward, next_observation, done]), [1, 5]))
                if total_steps > pre_train_steps:
                    if explore > endE:
                        explore -= stepDrop

                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.

                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        Q2 = sess.run(targetQN.Qout,
                                      feed_dict={targetQN.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        end_multiplier = -(trainBatch[:, 4] - 1)
                        doubleQ = Q2[range(batch_size), Q1]

                        targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, feed_dict={mainQN.imageIn: np.stack(trainBatch[:, 0] / 3.0),
                                                                    mainQN.targetQ: targetQ,
                                                                    mainQN.actions: trainBatch[:, 1]})
                        updateTarget(targetOps, sess)  # Update the target network toward the primary network.

                observation = next_observation
                reward_all += reward

                if done == True:
                    break

            myBuffer.add(episodeBuffer.buffer)
            reward_list.append(reward_all)

            # Periodically save the PPO_implementation.
            if i % 100 == 0:
                saver_new_model.save(sess, saving_path + '/model-agent_' + str(i) + '.ckpt')
                print(f"Saved Model at checkpoint {i}")

            # Print performance: total steps, rewards, current exploration rate
            if len(reward_list) % 10 == 0:
                print(total_steps, np.mean(reward_list[-10:]), explore)

            # Saves PPO_implementation if there is a new best score
            if np.mean(reward_list[-10:]) > current_best:
                current_best = np.mean(reward_list[-10:])
                saver_new_model.save(sess, saving_path + '/model-agent_' + str(i) + '.ckpt')
                print(f"New current best! That PPO_implementation is now saved!")


if __name__ == '__main__':

    main()