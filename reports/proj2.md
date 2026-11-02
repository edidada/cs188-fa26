# CS188 Fall 2026 Project 2 Multiagent Search

分支edidada/proj2，仅实现multiAgents.py。统一树搜索支持Pacman最大化、多个Ghost逐层最小化与均匀随机Ghost的Expectimax。深度仅在最后一个Ghost完成一整轮后增加；终局/深度界限使用指定评价函数。Alpha-beta保持原动作顺序，严格超出alpha/beta才剪枝，不在相等时剪枝，以遵守课程对扩展节点集合的要求。

Reflex与better评价考虑游戏得分、剩余食物与胶囊、最近食物距离、活动Ghost危险和可追到的受惊Ghost；胜负给予大幅奖励/惩罚。Reflex额外惩罚停止动作。不是用固定动作或读取测试答案替代搜索。

独立proj2.yml在Python3.11运行原无图形autograder，保留所有公开测试与随机游戏评分；要求每题不低于各题满分并允许官方奖励分。实际分数以完整Actions日志为准，本地未运行实验，未向课程平台提交。
