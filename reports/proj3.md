# CS188 Fall 2026 Project 3 Reinforcement Learning

分支edidada/proj3，实现同步Value Iteration、Prioritized Sweeping、表格Q-learning和线性近似Q-learning。同步迭代用新Counter保存整轮Bellman备份，避免同轮更新污染；终端值0，无动作策略None。Prioritized Sweeping以正概率转移构建前驱集合，优先处理绝对Bellman残差大的状态；仅超过theta时更新前驱优先级。

Q-learning使用奖励加折扣后继最大Q值作为目标；未见状态动作值0，epsilon-greedy探索，在并列最优动作间随机选择，终端状态不探索。近似Q采用特征与权重内积，先求完整TD误差再统一更新各权重，避免特征顺序改变目标。

策略分析题：零噪声允许安全通过Bridge；小discount偏近期近出口，大discount偏远出口；noise鼓励避开悬崖；足够大的正living reward使持续生存优于退出。有限50轮中无法仅靠固定epsilon和learning rate保证跨桥学习达99%（存在缺少关键探索的随机事件），故question7返回NOT POSSIBLE。

独立proj3.yml运行原全部无图形autograder，要求总分和逐题至少满分，允许官方奖励分；原测试、环境及评分器均不修改。本地未运行实验，不向课程平台提交。具体参数及学习表现以完整Actions为准。
