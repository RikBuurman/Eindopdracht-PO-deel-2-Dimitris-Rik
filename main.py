import random
from bke import MLAgent, is_winner, opponent, RandomAgent, train_and_plot

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward

random.seed(1)
 
my_agent = MyAgent(alpha=0.9, epsilon=0.1)
random_agent = RandomAgent()
 
train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=15,
    trainings=1000,
    validations=1000)