 #Past waarde   #hyperparameters1001000import random
import random
from bke import MLAgent, is_winner, opponent, RandomAgent, train_and_plot, plot_validation,validate

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1 #p
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward

random.seed(1)
 
my_agent = MyAgent(alpha=0.1, epsilon=0.1)
random_agent = RandomAgent()
 
train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=100,
    trainings=5000,
    validations=5000)

validation_agent = RandomAgent()
 
validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=100)

plot_validation(validation_result)