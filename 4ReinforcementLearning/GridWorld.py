import numpy as np
import random


'''
GridWorld类，用于生成一个网格世界，包含了若干个普通区域，若干个forbiddenArea，若干个target
本类借鉴了：https://github.com/ziwenhahaha/Code-of-RL-Beginning/的实现
转载请注明出处
'''
class GridWorld():
    # n行，m列，随机若干个forbiddenArea，随机若干个target
    # A1: move upwards
    # A2: move rightwards;
    # A3: move downwards;
    # A4: move leftwards;
    # A5: stay unchanged;

    stateMap = None  #大小为rows*columns的list，每个位置存的是state的编号
    RewardMap = None  #大小为rows*columns的list，每个位置存的是奖励值 0 1 -1
    Reward = 0             #targetArea的得分
    forbiddenAreaReward=0  #forbiddenArea的得分

    # desc 是自定义的地图，是一个二维数组，每个元素是一个字符，'#'表示forbiddenArea，'T'表示target
    # 例如：
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
        self.reward = reward
        self.forbiddenAreaReward = forbiddenAreaReward

        # 如果指定了desc，则使用desc初始化地图
        if (desc != None):
            self.rows = len(desc)
            self.columns = len(desc[0])
            l = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    tmp.append(forbiddenAreaReward if desc[i][j] == '#' else reward if desc[i][j] == 'T' else 0)
                l.append(tmp)
            self.rewardMap = np.array(l)
            self.stateMap = [
                [i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]
            return
        
        # 如果没有指定desc，则随机生成地图
        self.rows = rows
        self.columns = columns
        self.forbiddenAreaNums = forbiddenAreaNums
        self.targetNums = targetNums
        self.seed = seed

        random.seed(self.seed)

        # 生成随机地图
        # 生成一个随机的地图，地图大小为rows*columns，其中有forbiddenAreaNums个forbiddenArea，有targetNums个target
        l = [i for i in range(self.rows * self.columns)]
        random.shuffle(l)
        # 生成一个长度为rows*columns的list，每个元素是0，表示普通区域
        self.g = [0 for i in range(self.rows * self.columns)]
        # 向其中随机插入forbiddenArea和target
        for i in range(self.forbiddenAreaNums):
            self.g[l[i]] = forbiddenAreaReward
        for i in range(targetNums):
            self.g[l[self.forbiddenAreaNums + i]] = reward

        self.rewardMap = np.array(self.g).reshape(self.rows, self.columns)
        self.stateMap = [
            [i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]

    # 显示上述地图
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

    def getScore(self, nowState, action):
        # nowState是当前状态的编号，action是当前动作编号
        # 返回值是执行动作之后的状态编号和得分
        # 返回值格式：(score, nextState)
        nowx = nowState // self.columns
        nowy = nowState % self.columns
        # 先判断是否是边界
        if (nowx < 0 or nowx >= self.rows or nowy < 0 or nowy >= self.columns):
            print(f"Error: nowState is out of range: ({nowx}, {nowy})")
            return None
        if (action < 0 or action > 5):
            print(f"Error: action is out of range: {action}")
            return None

        # action: 上，右，下，左，不动
        # action: (x, y) e.g. (0, 1) 表示x不变，y+1
        action_list = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        nextx = nowx + action_list[action][0]
        nexty = nowy + action_list[action][1]
        # print(f"nowState: ({nowx}, {nowy}), action: {action}, nextState: ({nextx}, {nexty})")
        # 判断是否越界, 如果越界，则返回当前状态和-1分
        if (nextx < 0 or nextx >= self.rows or nexty < 0 or nexty >= self.columns):
            return (nowState, self.forbiddenAreaReward, False)
        
        reward = self.rewardMap[nextx][nexty]
        nextState = self.stateMap[nextx][nexty]
        # 表明到达了终点, 返回下一个状态和得分和是否结束
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
        policy是一个(rows*columns) * 5的矩阵，每一行表示一个状态，每一行的5个元素分别表示5个动作的概率, sum(policy[i]) = 1
        这个方法用于展示每个状态的每个动作的概率
        '''
        # 展示当前的策略
        s = ''
        # print(policy)
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
    def show_policy_list(self, policy):
        '''
        policy是一个(rows*columns) 的列表，每个元素代表在每个状态下的策略
        '''
        # 展示当前的策略
        s = ''
        print('现在的策略是：')
        for i in range(self.rows * self.columns):
            nowx = i // self.columns
            nowy = i % self.columns
            if self.rewardMap[nowx][nowy] == self.reward:
                s += '✅'
            if self.rewardMap[nowx][nowy] == 0:
                tmp = {0:"⬆️",1:"➡️",2:"⬇️",3:"⬅️",4:"🔄"}
                s += tmp[policy[i]]
            if self.rewardMap[nowx][nowy] == self.forbiddenAreaReward:
                tmp = {0:"⏫️",1:"⏩️",2:"⏬",3:"⏪",4:"🔄"} 
                s += tmp[policy[i]]
            if nowy == self.columns - 1:
                print(s)
                s = ''
    
    def get_final_state(self):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.rewardMap[i][j] == self.reward:
                    return i,j
                
    def reset(self):
        i = random.randint(0, self.rows-1)
        j = random.randint(0, self.columns-1)
        return self.stateMap[i][j]