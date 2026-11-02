# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for _ in range(self.iterations):
            updated = util.Counter()
            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if actions and not self.mdp.isTerminal(state):
                    updated[state] = max(self.computeQValueFromValues(state, action) for action in actions)
            self.values = updated


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        return sum(probability * (self.mdp.getReward(state, action, target) + self.discount * self.values[target])
                   for target, probability in self.mdp.getTransitionStatesAndProbs(state, action))

    def computeActionFromValues(self, state):
        actions = self.mdp.getPossibleActions(state)
        if not actions or self.mdp.isTerminal(state):
            return None
        return max(actions, key=lambda action: self.computeQValueFromValues(state, action))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        predecessors = {state: set() for state in states}
        queue = util.PriorityQueue()
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for target, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    if probability > 0:
                        predecessors[target].add(state)
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            if actions and not self.mdp.isTerminal(state):
                target = max(self.computeQValueFromValues(state, action) for action in actions)
                queue.push(state, -abs(self.values[state] - target))
        for _ in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            actions = self.mdp.getPossibleActions(state)
            self.values[state] = max(self.computeQValueFromValues(state, action) for action in actions)
            for previous in predecessors[state]:
                actions = self.mdp.getPossibleActions(previous)
                if actions and not self.mdp.isTerminal(previous):
                    target = max(self.computeQValueFromValues(previous, action) for action in actions)
                    difference = abs(self.values[previous] - target)
                    if difference > self.theta:
                        queue.update(previous, -difference)

