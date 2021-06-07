from environments.snake_env import SnakeEnvironment
import os
from dddqn_model.dddqn import *

saving_path = "models_selfplay_2"  # The path to save our PPO_implementation to.
load_model_path = "models_selfplay_2"

batch_size = 512  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 0.1  # Starting chance of random action
endE = 0.0001  # Final chance of random action
annealing_steps = 500000.  # How many steps of training to reduce startE to endE.
num_episodes = 200  # How many episodes of game environment to train network with.
pre_train_steps = 5000  # How many steps of random actions before training begins.
max_epLength = 10000000  # The max amount of steps in an episode.
load_model = False  # Whether to load a saved PPO_implementation.
h_size = 1296 * 2  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network


def main():
    # If agent_1 reaches this threshold, its weights get copied to agent 2.
    threshold = -0.5

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
    agent_1_targetQN = Qnetwork(h_size=h_size, scope="target_agent_1")

    # Create Adversary Agent
    agent_2 = Qnetwork(h_size=h_size, scope="main_agent_2")

    # Tensorflow restore weight setting
    init = tf.compat.v1.global_variables_initializer()

    # Trainable variable for pretrain agent, agent1, agent2
    trainables = tf.compat.v1.trainable_variables()

    weights_agent_1 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_1', 'target_agent_1']]

    weights_agent_2 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_2']]

    # Copy weight of agent1 to agent2
    update_weights = [tf.compat.v1.assign(weights_2, weights_1) for (weights_2, weights_1) in
                      zip(weights_agent_2, weights_agent_1)]

    # Tensorflow Saver for agent2
    saver_new_model = tf.compat.v1.train.Saver(weights_agent_1)

    # Trainable variable for target network
    targetOps_new = updateTargetGraph(weights_agent_1, tau)

    # Set the rate of random action decrease.
    exploration = startE
    stepDrop = (startE - endE) / annealing_steps

    # Create lists to contain total rewards and steps per episode
    reward_list_agent_2 = []
    reward_list_agent_1 = []

    # Make a path for our PPO_implementation to be saved in.
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(update_weights)

        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(load_model_path)
            saver_new_model.restore(sess, ckpt.model_checkpoint_path)
            sess.run(update_weights)
            print('Loading Saving Model...@:')
            print(ckpt.model_checkpoint_path)


        # Buffer for saving winning agent history
        myBuffer = experience_buffer()
        for i in range(num_episodes):


            # Decay exploration parameter
            if exploration > endE:
                exploration -= stepDrop

            # Matching agent1, agent2 for Self-Play
            for k in range(0, 100):
                episodeBuffer = [experience_buffer(), experience_buffer()]

                # Reset environment and get first new observation
                obs = env.reset()

                # Initialize state, reward, end flag of agent1, agent2
                obs_agent_1 = obs[1]
                obs_agent_2 = obs[0]


                # Initialize sum of reward of agent1, agent2
                sum_rewards_agent_2 = 0
                sum_rewards_agent_1 = 0

                # The Q-Network
                for current_step in range(0, max_epLength):

                    # Select action of agent 2
                    if np.random.rand(1) < exploration or i < 10:
                        action_agent_2 = np.random.randint(0, 4)
                    else:
                        action_agent_2 = sess.run(agent_2.predict, feed_dict={agent_2.imageIn: [obs_agent_2 / 3.0]})[
                            0]

                    # Select action of agent 1
                    if np.random.rand(1) < exploration or i < 10:
                        action_agent_1 = np.random.randint(0, 4)
                    else:
                        action_agent_1 = sess.run(agent_1.predict, feed_dict={agent_1.imageIn: [obs_agent_1 / 3.0]})[
                            0]

                    # Move agent1, agent2 and get reward, next state, end flag of each agent and end flag of current episode
                    next_state, reward, done, d_common = env.step([action_agent_2, action_agent_1])

                    # Reward of agent1, agent2
                    reward_agent_2 = reward[0]
                    reward_agent_1 = reward[1]

                    # End flag of agent1, agent2
                    done_agent_2 = done[0]
                    done_agent_1 = done[1]

                    # Next state of agent1, agent2
                    next_state_agent_2 = next_state[0]
                    next_state_agent_1 = next_state[1]


                    # Save history of agent1, agent2 if there are not dead
                    if done_agent_2 == False:
                        episodeBuffer[1].add(
                            np.reshape(np.array(
                                [obs_agent_2, action_agent_2, reward_agent_2, next_state_agent_2, done_agent_2]),
                                       [1, 5]))
                    if done_agent_1 == False:
                        episodeBuffer[0].add(
                            np.reshape(np.array(
                                [obs_agent_1, action_agent_1, reward_agent_1, next_state_agent_1, done_agent_1]),
                                       [1, 5]))

                    # Add a reward of each agent to total reward
                    sum_rewards_agent_2 += reward_agent_2
                    sum_rewards_agent_1 += reward_agent_1

                    # Save a next state to current state for next step
                    obs_agent_2 = next_state_agent_2
                    obs_agent_1 = next_state_agent_1

                    # If either of the agent is done, give a win to the other agent
                    # Agent 1 wins
                    if (done_agent_2 == True):
                        break
                    # Agent 2 wins
                    elif (done_agent_1 == True):
                        break


                # Start training if size of winning agent buffer is large than batch size
                if len(myBuffer.buffer) > batch_size * 16:

                    # Repeat a training 16 times
                    for q in range(0, 16):
                        trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.

                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(agent_1.predict,
                                      feed_dict={agent_1.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        Q2 = sess.run(agent_1_targetQN.Qout,
                                      feed_dict={agent_1_targetQN.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        end_multiplier = -(trainBatch[:, 4] - 1)
                        doubleQ = Q2[range(batch_size), Q1]

                        targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)

                        # Update the network with our target values.
                        _ = sess.run(agent_1.updateModel,
                                     feed_dict={agent_1.imageIn: np.stack(trainBatch[:, 0] / 3.0),
                                                agent_1.targetQ: targetQ,
                                                agent_1.actions: trainBatch[:, 1]})
                        updateTarget(targetOps_new, sess)  # Update the target network toward the primary network.

                # Save a history of agent_1 to buffer
                myBuffer.add(episodeBuffer[0].buffer)

                # Save sum of each agent for printing performance
                reward_list_agent_2.append(sum_rewards_agent_2)
                reward_list_agent_1.append(sum_rewards_agent_1)

            # The self-play algorithm.
            if np.mean(reward_list_agent_1[-10:]) >= threshold and np.mean(reward_list_agent_1[-10:]) > np.mean(
                    reward_list_agent_2[-10:]):
                print('Updating Weight...')
                sess.run(update_weights)
                myBuffer = experience_buffer()
                threshold += 0.2
                print(f'The new threshold is {threshold}')
                saver_new_model.save(sess, saving_path + '/model-agent_' + str(i) + '.ckpt')
                print("Model Saved")

            # Additional save point
            if len(reward_list_agent_1) > 20:

                if np.mean(reward_list_agent_1[-20:]) > np.mean(reward_list_agent_2[-20:]):
                    print('Updating Weight...')
                    saver_new_model.save(sess, saving_path + '/model-agent_' + str(i) + '.ckpt')
                    print("Saved Model")

            # Periodically save the PPO_implementation
            if i % 100 == 0:
                saver_new_model.save(sess, saving_path + '/model-agent_' + str(i) + '.ckpt')
                print("Saved Model")


            # Periodically print performance of agents
            if len(reward_list_agent_1) % 50 == 0:
                print(i, "agent_2", np.mean(reward_list_agent_2[-10:]), exploration)
                print(i, "agent_1", np.mean(reward_list_agent_1[-10:]), exploration)
                print("")


if __name__ == '__main__':
    main()
