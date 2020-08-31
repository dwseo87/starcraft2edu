import pandas as pd
import numpy as np

# reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        #dataframe 대신 numpy배열을 쓰는 경우도 많다
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            # state_action = self.q_table.ix[observation, :]
            state_action = self.q_table.loc[observation, :]

            # some actions have the same value
            # permutation -> 순열을 바꾸고 random하게 선택하도록 하는 부분
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # q_predict = self.q_table.ix[s, a]
        q_predict = self.q_table.loc[s, a]
        # q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        # update
        # self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))