from environments.team_snake_env import SnakeEnvironment
from dddqn_model.dddqn import *
import time

#################
#Enter preferred checkpoint here
checkpoint = None
###################

if checkpoint == None:
    print('You need to enter an eligible checkpoint')

batch_size = 512  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 0.1  # Starting chance of random action
endE = 0.0001  # Final chance of random action
annealing_steps = 500000.  # How many steps of training to reduce startE to endE.
num_episodes = 5000000  # How many episodes of game environment to train network with.
pre_train_steps = 5000  # How many steps of random actions before training begins.
max_epLength = 5000000  # The max allowed length of our episode.
load_model = True  # Whether to load a saved PPO_implementation.
h_size = 1296 * 2  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

def main():


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

    agent_3 = Qnetwork(h_size=h_size, scope="main_agent_3")

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


    # Create lists to contain total rewards and steps per episode
    reward_list_agent_4 = []
    reward_list_agent_3 = []
    reward_list_agent_2 = []
    reward_list_agent_1 = []

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver_new_model_1.restore(sess, f'./models_team\model-agent_1_{checkpoint}.ckpt')
        saver_new_model_3.restore(sess, f'./models_team\model-agent_3_{checkpoint}.ckpt')

        sess.run(update_weights_agent_2)
        sess.run(update_weights_agent_4)

        print('Loading Model...:')

        for i in range(num_episodes):
            # Reset winning number
            win_num = [0, 0]


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

                # Initialize sum of reward of agent1, agent2
                sum_rewards_agent_4 = 0
                sum_rewards_agent_3 = 0
                sum_rewards_agent_2 = 0
                sum_rewards_agent_1 = 0

                # The Q-Network
                for j in range(0, max_epLength):
                    env.render()
                    time.sleep(0.1)

                    # Select action of agent4

                    action_agent_4 = sess.run(agent_4.predict, feed_dict={agent_4.imageIn: [obs_agent_4 / 3.0]})[
                        0]

                    # Select action of agent3
                    action_agent_3 = sess.run(agent_3.predict, feed_dict={agent_3.imageIn: [obs_agent_3 / 3.0]})[
                        0]

                    # Select action of agent2
                    action_agent_2 = sess.run(agent_2.predict, feed_dict={agent_2.imageIn: [obs_agent_2 / 3.0]})[
                        0]

                    # Select action of agent1
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

                # Save sum of each agent for printing performance
                reward_list_agent_4.append(sum_rewards_agent_4)
                reward_list_agent_3.append(sum_rewards_agent_3)
                reward_list_agent_2.append(sum_rewards_agent_2)
                reward_list_agent_1.append(sum_rewards_agent_1)


            # Periodically print performance of agents "step, most current mean reward"
            if len(reward_list_agent_1) % 10 == 0:
                print(i, "agent_4", np.mean(reward_list_agent_4[-10:]))
                print(i, "agent_3", np.mean(reward_list_agent_3[-10:]))
                print(i, "agent_2", np.mean(reward_list_agent_2[-10:]))
                print(i, "agent_1", np.mean(reward_list_agent_1[-10:]))
                print("")


if __name__ == '__main__':
    main()