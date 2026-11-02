# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successor = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successor) - (2 if action == Directions.STOP else 0)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        return _tree_action(self, gameState, 'minimax')

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        return _tree_action(self, gameState, 'alphabeta')

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        return _tree_action(self, gameState, 'expectimax')

def betterEvaluationFunction(currentGameState: GameState):
    if currentGameState.isWin():
        return currentGameState.getScore() + 100000
    if currentGameState.isLose():
        return currentGameState.getScore() - 100000
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    value = currentGameState.getScore() - 4 * len(food) - 12 * len(currentGameState.getCapsules())
    if food:
        value += 12.0 / (1 + min(manhattanDistance(position, dot) for dot in food))
    for ghost in currentGameState.getGhostStates():
        distance = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > distance:
            value += 200.0 / (distance + 1)
        elif not ghost.scaredTimer:
            if distance <= 1:
                value -= 1000
            else:
                value -= 5.0 / distance
    return value

# Abbreviation
better = betterEvaluationFunction


def _tree_action(agent, initial, mode):
    """Search complete Pacman/ghost rounds with the original successor order."""
    def visit(state, index, rounds, alpha, beta):
        if rounds == agent.depth or state.isWin() or state.isLose():
            return agent.evaluationFunction(state), None
        actions = state.getLegalActions(index)
        if not actions:
            return agent.evaluationFunction(state), None
        next_index = (index + 1) % state.getNumAgents()
        next_rounds = rounds + (next_index == 0)
        choice = actions[0]
        value = -float('inf') if index == 0 else float('inf')
        if index != 0 and mode == 'expectimax':
            value = sum(visit(state.generateSuccessor(index, action), next_index, next_rounds, alpha, beta)[0]
                        for action in actions) / len(actions)
            return value, choice
        for action in actions:
            child = state.generateSuccessor(index, action)
            candidate = visit(child, next_index, next_rounds, alpha, beta)[0]
            if index == 0:
                if candidate > value:
                    value, choice = candidate, action
                if mode == 'alphabeta':
                    if value > beta:
                        return value, choice
                    alpha = max(alpha, value)
            else:
                if candidate < value:
                    value, choice = candidate, action
                if mode == 'alphabeta':
                    if value < alpha:
                        return value, choice
                    beta = min(beta, value)
        return value, choice
    return visit(initial, 0, 0, -float('inf'), float('inf'))[1] or Directions.STOP
