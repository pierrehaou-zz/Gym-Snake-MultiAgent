from train_dddqn_single import *

load_model_path = 'pretrained_models/pretrained_models_single'
h_size = 1296 * 2  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
max_epLength = 10000000  # The max allowed length of our episode.
num_episodes = 200  # How many episodes of game environment to train network with.


def main():
    spacing = 22
    dimensions = 15
    history = 4
    env = SnakeEnvironment(num_agents=1, num_fruits=3, spacing=spacing, dimensions=dimensions, flatten_states=False,
                           reward_killed=-1.0, history=history)
    env.reset()

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    mainQN = Qnetwork(h_size=h_size, scope="main_new")

    init = tf.compat.v1.global_variables_initializer()
    load_prev_model = tf.compat.v1.train.Saver()

    # create lists to contain total rewards and steps per episode
    reward_list = []
    total_steps = 0

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(load_model_path)
        load_prev_model.restore(sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        for i in range(num_episodes):

            # Reset environment and get first new observation
            state = env.reset()
            reward_all = 0
            current_step = 0

            # The Q-Network
            while current_step < max_epLength:
                env.render()
                current_step += 1

                # Choose an action
                a = sess.run(mainQN.predict, feed_dict={mainQN.imageIn: [state / 3.0]})[0]
                next_state, reward, _, done = env.step(a)
                total_steps += 1

                state = next_state
                reward_all += reward

                if done == True:
                    break

            reward_list.append(reward_all)

            # Occasionally print total_steps and average reward
            if len(reward_list) % 10 == 0:
                print(total_steps, np.mean(reward_list[-10:]))


if __name__ == '__main__':
    main()
