import numpy as np
import random
import matplotlib.pyplot as plt

'''
GridWorld: used to generate a grid world, including several normal areas, several forbidden areas, and several targets
This class is inspired by: https://github.com/ziwenhahaha/Code-of-RL-Beginning/

!!! Please credit the source when reposting. !!!
'''

class GridWorld():
    # n rows, m columns, several forbiddenArea, several target
    # A1: move upwards
    # A2: move rightwards;
    # A3: move downwards;
    # A4: move leftwards;
    # A5: stay unchanged;

    # desc is a custom map, it is a two-dimensional array, each element is a character, '#' represents forbiddenArea, 'T' represents target
    # For example：
    #     desc = [
    #     ['.', '.', 'T', '.', '.'],
    #     ['.', '#', '.', '#', '.'],
    #     ['.', '.', '.', '.', '.'],
    #     ['#', '.', '.', '.', '.']
    # ]
    def __init__(self,
                rows = 4,
                columns = 5,
                forbiddenAreaNums = 3,
                targetNums = 1,
                seed = -1,
                reward = 1,
                forbiddenAreaReward = -1,
                desc = None):
        
        # The reward of the target
        self.reward = reward
        # The penalty of the forbidden area
        self.forbiddenAreaReward = forbiddenAreaReward

        # if desc is not None, then use the custom map
        if (desc != None):
            self.rows = len(desc)
            self.columns = len(desc[0])
            l = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    tmp.append(forbiddenAreaReward if desc[i][j] == '#' else reward if desc[i][j] == 'T' else 0)
                l.append(tmp)
            # rewardMap is a two-dimensional array, each element is a number, 0 represents normal area, forbiddenAreaReward represents forbidden area, reward represents target
            self.rewardMap = np.array(l)
            # stateMap is a two-dimensional array, each element is a number, which represents the state number of the corresponding position
            self.stateMap = [
                [i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]
            return
        
        # if desc is None, then generate a random map
        self.rows = rows
        self.columns = columns
        self.forbiddenAreaNums = forbiddenAreaNums
        self.targetNums = targetNums
        self.seed = seed

        # random.seed(self.seed)

        # Generate a random map, the size of the map is rows*columns, there are forbiddenAreaNums forbidden areas, and there are targetNums targets

        l = [i for i in range(self.rows * self.columns)]
        random.shuffle(l)
        
        # g is a one-dimensional array, each element is a number, 0 represents normal area, forbiddenAreaReward represents forbidden area, reward represents target
        self.g = [0 for i in range(self.rows * self.columns)]

        # Randomly select forbiddenAreaNums forbidden areas and targetNums targets
        for i in range(self.forbiddenAreaNums):
            self.g[l[i]] = forbiddenAreaReward
        for i in range(targetNums):
            self.g[l[self.forbiddenAreaNums + i]] = reward

        self.rewardMap = np.array(self.g).reshape(self.rows, self.columns)
        self.stateMap = [
            [i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]

    # Show the map
    def show(self):
        for i in range(self.rows):
            s = ''
            for j in range(self.columns):
                temp = {
                    0: "⬜️",
                    self.forbiddenAreaReward: "🚫",
                    self.reward: "✅"
                }
                s += temp[self.rewardMap[i][j]]
            print(s)

    def step(self, nowState, action):
        '''
        Return the next state and the score after taking the action and whether the game is over: (nextState, reward, isEnd)
        nowState: the number of the current state, action: the number of the current action
        action: 0: up, 1: right, 2: down, 3: left, 4: stay unchanged

        '''
        nowx = nowState // self.columns
        nowy = nowState % self.columns
        # Whether the current state is out of range
        if (nowx < 0 or nowx >= self.rows or nowy < 0 or nowy >= self.columns):
            print(f"Error: nowState is out of range: ({nowx}, {nowy})")
            return None
        # Whether the current action is out of range
        if (action < 0 or action > 5):
            print(f"Error: action is out of range: {action}")
            return None

        # action: left, up, right, down, stay unchanged
        # action: (x, y) e.g. (0, 1) represents x unchanged, y + 1
        action_list = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        nextx = nowx + action_list[action][0]
        nexty = nowy + action_list[action][1]
        # print(f"nowState: ({nowx}, {nowy}), action: {action}, nextState: ({nextx}, {nexty})")
        # Whether the next state is out of range
        if (nextx < 0 or nextx >= self.rows or nexty < 0 or nexty >= self.columns):
            return (nowState, self.forbiddenAreaReward, False)
        
        reward = self.rewardMap[nextx][nexty]
        nextState = self.stateMap[nextx][nexty]
        # Whether the game is over
        if reward == self.reward:
            return nextState, reward, True
        return nextState, reward, False     
    
    def get_episode_score(self, now_state, now_action, policy, steps = None, stop_when_reach_target = False):
        # now_state是当前状态的编号，action是当前动作编号
        # policy是一个(rows*columns) * 5的矩阵，每一行表示一个状态，每一行的5个元素分别表示5个动作的概率, sum(policy[i]) = 1

        res = []

        if stop_when_reach_target:
            while True:
                reward, next_state = self.getScore(now_state, now_action)
                # 注意下这里的policy选择是根据概率选择的
                next_action = np.random.choice(range(5), size=1, replace=False, p=policy[next_state])[0]
                # 如果到达了目标，就结束
                if reward == self.reward:
                    next_state = now_state
                    next_action = now_action
                    res.append((now_state, now_action, reward, next_state, next_action))
                    break
                res.append((now_state, now_action, reward, next_state, next_action))
                now_state = next_state
                now_action = next_action
            return res


        for i in range(steps+1):
            reward, next_state = self.getScore(now_state, now_action)
            # policy[next_state] 应该是一个表示策略的数组，长度为5，对应于从状态 next_state 出发的每个动作的选择概率
            next_action = np.random.choice(range(5), size=1, replace=False, p=policy[next_state])[0]

            res.append((now_state, now_action, reward, next_state, next_action))

            now_state = next_state
            now_action = next_action
        return res
    
    def show_policy_matirx(self, policy):
        '''
        policy: a (rows*columns) * 5 matrix, each row represents a state, and the 5 elements of each row represent the probabilities of 5 actions, sum(policy[i]) = 1
        This method is used to display the probability of each action of each state
        '''
        s = ''
        print(f'Now policy:')
        for i in range(self.rows * self.columns):
            nowx = i // self.columns
            nowy = i % self.columns
            if self.rewardMap[nowx][nowy] == self.reward:
                s += '✅'
            if self.rewardMap[nowx][nowy] == 0:
                tmp = {0:"⬆️",1:"➡️",2:"⬇️",3:"⬅️",4:"🔄"}
                s += tmp[np.argmax(policy[i])]
            if self.rewardMap[nowx][nowy] == self.forbiddenAreaReward:
                tmp = {0:"⏫️",1:"⏩️",2:"⏬",3:"⏪",4:"🔄"} 
                s += tmp[np.argmax(policy[i])]
            if nowy == self.columns - 1:
                print(s)
                s = ''
    
    def get_final_state(self):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.rewardMap[i][j] == self.reward:
                    return i,j
                
    def show_state_value(self, state_values):

        fig, ax = plt.subplots()

        cax = ax.matshow(state_values, cmap='viridis')

        for i in range(state_values.shape[0]):
            for j in range(state_values.shape[1]):
                ax.text(j, i, f'{state_values[i, j]:.1f}', va='center', ha='center', color='white')

        plt.show()