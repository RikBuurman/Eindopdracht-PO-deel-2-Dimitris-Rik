import random
from bke import MLAgent, RandomAgent, is_winner,opponent, plot_validation, train, train_and_validate, validate


# Class voor slimme agent (MyAgent)
class MyAgent(MLAgent):  
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1  # Kent score 1 toe bij winst tegen random tegenstander
        elif is_winner(board, opponent[self.symbol]):
            reward = -1  # Kent score -1 toe bij verlies tegen random tegenstander
        else:
            reward = 0  # Kent score 0 toe bij gelijkspel tegen random tegenstander
        return reward

random.seed(1) # Begingetal patroon (seed) vastzetten voor betere vergelijking

# Hyperparameters van de slimme agent:
# Hoe snel pakt de agent nieuwe kennis op: alpha, 
# En hoe snel probeert de agent willekeurige acties: epsilon
my_agent = MyAgent(alpha=0.1, epsilon=0.1) 

random_agent = RandomAgent() 

# Functie voor trainen van agent
train_and_validate(  
    agent=my_agent,
    validation_agent=random_agent,  # Tegenstander is de random agent
    iterations=50,  # Aantal herhalingen trainproces
    trainings=500,  # Aantal trainingen per herhaling
    validations=1000)  # Aantal validatiespellen per herhaling

# Einde trainingsfase slimme agent, nu naar validatie
my_agent.learning = False 

validation_agent = RandomAgent()

# Functie voor het valideren van de slimme agent
validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=1000)

# 'tekenen' pie chart met resultaten validatie
plot_validation(validation_result) 