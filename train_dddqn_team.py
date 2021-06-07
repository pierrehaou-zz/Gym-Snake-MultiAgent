from environments.team_snake_env import SnakeEnvironment
import os
from dddqn_model.dddqn import *

saving_path = "models_team"  # The path to save our PPO_implementation to.
load_model_path = "models_team"

batch_size = 512  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 0.1  # Starting chance of random action
endE = 0.0001  # Final chance of random action
annealing_steps = 500000.  # How many steps of training to reduce startE to endE.
num_episodes = 5000000  # How many episodes of game environment to train network with.
pre_train_steps = 5000  # How many steps of random actions before training begins.
max_epLength = 5000000  # The max allowed length of our episode.
load_model = False  # Whether to load a saved PPO_implementation.
h_size = 1296 * 2  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

def main():
    # If blue team reaches this threshold, its weights get copied to other team. Will gradually increase.
    threshold = -0.5

    # Environment setting
    spacing = 22
    dimensions = 15
    history = 4
    env = SnakeEnvironment(num_agents=4, num_fruits=3, spacing=spacing, dimensions=dimensions, flatten_states=False,
                           reward_killed=-1.0, history=history)
    env.reset()

    # Tensorflow PPO_implementation setting
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Create Blue Team
    agent_1 = Qnetwork(h_size=h_size, scope="main_agent_1")
    agent_1_targetQN = Qnetwork(h_size=h_size, scope="target_agent_1")

    agent_3 = Qnetwork(h_size=h_size, scope="main_agent_3")
    agent_3_targetQN = Qnetwork(h_size=h_size, scope="target_agent_3")

    # Create Adversary agents
    agent_2 = Qnetwork(h_size=h_size, scope="main_agent_2")

    agent_4 = Qnetwork(h_size=h_size, scope="main_agent_4")


    # Tensorflow restore weight setting
    init = tf.compat.v1.global_variables_initializer()

    # Trainable variable for pretrain agent, agent1, agent2
    trainables = tf.compat.v1.trainable_variables()

    weights_agent_1 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_1', 'target_agent_1']]
    weights_agent_3 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_3', 'target_agent_3']]


    weights_agent_2 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_2']]
    weights_agent_4 = [v for v in trainables if v.name.split('/')[0] in ['main_agent_4']]


    # Copy weight of agent1 to agent2

    update_weights_agent_2 = [tf.compat.v1.assign(weights_2, weights_1) for (weights_2, weights_1) in
                      zip(weights_agent_2, weights_agent_1)]

    update_weights_agent_4 = [tf.compat.v1.assign(weights_4, weights_3) for (weights_4, weights_3) in
                      zip(weights_agent_4, weights_agent_3)]


    saver_new_model_1 = tf.compat.v1.train.Saver(weights_agent_1)
    saver_new_model_3 = tf.compat.v1.train.Saver(weights_agent_3)


    # Trainable variable for target network
    targetOps_new_1 = updateTargetGraph(weights_agent_1, tau)
    targetOps_new_3 = updateTargetGraph(weights_agent_3, tau)


    # Set the rate of random action decrease.
    exploration = startE
    stepDrop = (startE - endE) / annealing_steps

    # Create lists to contain total rewards and steps per episode
    reward_list_agent_4 = []
    reward_list_agent_3 = []
    reward_list_agent_2 = []
    reward_list_agent_1 = []

    # Make a path for our PPO_implementation to be saved in.
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(update_weights_agent_2)
        sess.run(update_weights_agent_4)


        # Buffer for saving team's agent history
        myBuffer_1 = experience_buffer()
        myBuffer_3 = experience_buffer()
        for i in range(num_episodes):

            # Decay exploration parameter
            if exploration > endE:
                exploration -= stepDrop

            # Matching agent1, agent2, agent3, agent 4 for Self-Play
            for k in range(0, 100):
                episodeBuffer = [experience_buffer(), experience_buffer(), experience_buffer(), experience_buffer()]

                # Reset environment and get first new observation
                obs = env.reset()

                # Initialize state, reward, end flag of agent1, agent2
                obs_agent_1 = obs[0]
                obs_agent_2 = obs[1]
                obs_agent_3 = obs[2]
                obs_agent_4 = obs[3]


                # Initialize winning number for Self-Play
                win_index = None

                # Initialize sum of reward of agent1, agent2
                sum_rewards_agent_4 = 0
                sum_rewards_agent_3 = 0
                sum_rewards_agent_2 = 0
                sum_rewards_agent_1 = 0

                # The Q-Network
                for j in range(0, max_epLength):

                    # Select action of agent4
                    if np.random.rand(1) < exploration or i < 10:
                        action_agent_4 = np.random.randint(0, 4)
                    else:
                        action_agent_4 = sess.run(agent_4.predict, feed_dict={agent_4.imageIn: [obs_agent_4 / 3.0]})[
                            0]

                    # Select action of agent3
                    if np.random.rand(1) < exploration or i < 10:
                        action_agent_3 = np.random.randint(0, 4)
                    else:
                        action_agent_3 = sess.run(agent_3.predict, feed_dict={agent_3.imageIn: [obs_agent_3 / 3.0]})[
                            0]

                    # Select action of agent2
                    if np.random.rand(1) < exploration or i < 10:
                        action_agent_2 = np.random.randint(0, 4)
                    else:
                        action_agent_2 = sess.run(agent_2.predict, feed_dict={agent_2.imageIn: [obs_agent_2 / 3.0]})[
                            0]

                    # Select action of agent1
                    if np.random.rand(1) < exploration or i < 10:
                        action_agent_1 = np.random.randint(0, 4)
                    else:
                        action_agent_1 = sess.run(agent_1.predict, feed_dict={agent_1.imageIn: [obs_agent_1 / 3.0]})[
                            0]

                    # Move agent1, agent2 and get reward, next state, end flag of each agent and end flag of current episode
                    next_obs, reward, done, d_common = env.step([action_agent_1, action_agent_2, action_agent_3, action_agent_4])

                    reward_agent_1 = reward[0]
                    reward_agent_2 = reward[1]
                    reward_agent_3 = reward[2]
                    reward_agent_4 = reward[3]

                    d_agent_1 = done[0]
                    d_agent_2 = done[1]
                    d_agent_3 = done[2]
                    d_agent_4 = done[3]

                    next_obs_agent_1 = next_obs[0]
                    next_obs_agent_2 = next_obs[1]
                    next_obs_agent_3 = next_obs[2]
                    next_obs_agent_4 = next_obs[3]


                    # Save history of agents if they are not dead
                    if d_agent_4 == False:
                        episodeBuffer[3].add(
                            np.reshape(
                                np.array([obs_agent_4, action_agent_4, reward_agent_4, next_obs_agent_4, d_agent_4]),
                                [1, 5]))
                    if d_agent_3 == False:
                        episodeBuffer[2].add(
                            np.reshape(
                                np.array([obs_agent_3, action_agent_3, reward_agent_3, next_obs_agent_3, d_agent_3]),
                                [1, 5]))
                    if d_agent_2 == False:
                        episodeBuffer[1].add(
                            np.reshape(
                                np.array([obs_agent_2, action_agent_2, reward_agent_2, next_obs_agent_2, d_agent_2]),
                                [1, 5]))
                    if d_agent_1 == False:
                        episodeBuffer[0].add(
                            np.reshape(
                                np.array([obs_agent_1, action_agent_1, reward_agent_1, next_obs_agent_1, d_agent_1]),
                                [1, 5]))

                    sum_rewards_agent_4 += reward_agent_4
                    sum_rewards_agent_3 += reward_agent_3
                    sum_rewards_agent_2 += reward_agent_2
                    sum_rewards_agent_1 += reward_agent_1

                    obs_agent_4 = next_obs_agent_4
                    obs_agent_3 = next_obs_agent_3
                    obs_agent_2 = next_obs_agent_2
                    obs_agent_1 = next_obs_agent_1

                    # Blue team wins
                    if (d_agent_2 == True and d_agent_4 == True):

                        break

                    # Adversary team wins
                    elif (d_agent_1 == True and d_agent_3 == True):

                        break

                # Start training if size of winning agent buffer is large than batch size
                if len(myBuffer_1.buffer) > batch_size * 16:

                    # Repeat a training 16 times
                    for q in range(0, 16):
                        trainBatch = myBuffer_1.sample(batch_size)  # Get a random batch of experiences.

                        # Below we perform the Double-DQN update to the target Q-values
                        end_multiplier = -(trainBatch[:, 4] - 1)

                        Q1_agent_1 = sess.run(agent_1.predict,
                                      feed_dict={agent_1.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        Q2_agent_1 = sess.run(agent_1_targetQN.Qout,
                                      feed_dict={agent_1_targetQN.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        doubleQ_agent_1 = Q2_agent_1[range(batch_size), Q1_agent_1]

                        targetQ_agent_1 = trainBatch[:, 2] + (y * doubleQ_agent_1 * end_multiplier)


                        # Update the network with our target values.
                        _ = sess.run(agent_1.updateModel,
                                     feed_dict={agent_1.imageIn: np.stack(trainBatch[:, 0] / 3.0),
                                                agent_1.targetQ: targetQ_agent_1,
                                                agent_1.actions: trainBatch[:, 1]})
                        updateTarget(targetOps_new_1, sess)  # Update the target network toward the primary network.

                # Save a history of agent_1 to buffer
                myBuffer_1.add(episodeBuffer[0].buffer)


                if len(myBuffer_3.buffer) > batch_size * 16:

                    for q in range(0, 16):
                        trainBatch = myBuffer_3.sample(batch_size)  # Get a random batch of experiences.

                        # Below we perform the Double-DQN update to the target Q-values
                        end_multiplier = -(trainBatch[:, 4] - 1)

                        # Double-DQN update for agent 3
                        Q1_agent_3 = sess.run(agent_1.predict,
                                              feed_dict={agent_1.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        Q2_agent_3 = sess.run(agent_3_targetQN.Qout,
                                              feed_dict={agent_3_targetQN.imageIn: np.stack(trainBatch[:, 3] / 3.0)})
                        doubleQ_agent_3 = Q2_agent_3[range(batch_size), Q1_agent_3]

                        targetQ_agent_3 = trainBatch[:, 2] + (y * doubleQ_agent_3 * end_multiplier)

                        _ = sess.run(agent_3.updateModel,
                                     feed_dict={agent_3.imageIn: np.stack(trainBatch[:, 0] / 3.0),
                                                agent_3.targetQ: targetQ_agent_3,
                                                agent_3.actions: trainBatch[:, 1]})
                        updateTarget(targetOps_new_3, sess)


                myBuffer_3.add(episodeBuffer[2].buffer)



                # Save sum of each agent for printing performance
                reward_list_agent_4.append(sum_rewards_agent_4)
                reward_list_agent_3.append(sum_rewards_agent_3)
                reward_list_agent_2.append(sum_rewards_agent_2)
                reward_list_agent_1.append(sum_rewards_agent_1)


            # self play algorithm
            if np.mean(reward_list_agent_1[-10:] + reward_list_agent_3[-10:]) >= threshold and np.mean(reward_list_agent_1[-10:] + reward_list_agent_3[-10:]) > np.mean(
                    reward_list_agent_2[-10:] + reward_list_agent_4[-10:]):
                print('Updating Weight...')
                sess.run(update_weights_agent_2)
                sess.run(update_weights_agent_4)
                threshold += 0.2
                print(f'The new threshold is {threshold}')

                saver_new_model_1.save(sess, saving_path + '/model-agent_1_' + str(i) + '.ckpt')
                saver_new_model_3.save(sess, saving_path + '/model-agent_3_' + str(i) + '.ckpt')
                print("Saved Model")

                myBuffer_1 = experience_buffer()
                myBuffer_3 = experience_buffer()

            # Alternative self play algorithm
            if len(reward_list_agent_1) > 20:

                if np.mean(reward_list_agent_1[-20:] + reward_list_agent_3[-20:]) > np.mean(
                        reward_list_agent_2[-20:] + reward_list_agent_4[-20:]):
                    print('Updating Weight...')
                    sess.run(update_weights_agent_2)
                    sess.run(update_weights_agent_4)

                    saver_new_model_1.save(sess, saving_path + '/model-agent_1_' + str(i) + '.ckpt')
                    saver_new_model_3.save(sess, saving_path + '/model-agent_3_' + str(i) + '.ckpt')
                    print("Saved Model")

                    myBuffer_1 = experience_buffer()
                    myBuffer_3 = experience_buffer()


            # Periodically print performance of agents "step, most current mean reward, current explore rate"
            if len(reward_list_agent_1) % 10 == 0:
                print(i, "agent_4", np.mean(reward_list_agent_4[-10:]), exploration)
                print(i, "agent_3", np.mean(reward_list_agent_3[-10:]), exploration)
                print(i, "agent_2", np.mean(reward_list_agent_2[-10:]), exploration)
                print(i, "agent_1", np.mean(reward_list_agent_1[-10:]), exploration)
                print("")


if __name__ == '__main__':
    main()