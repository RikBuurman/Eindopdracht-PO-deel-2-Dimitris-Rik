import random
from bke import MLAgent, is_winner, opponent, RandomAgent, plot_validation, validate, train_and_validate, train

class MyAgent(MLAgent): #Class voor slimme agent (MyAgent)
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1 #kent score 1 toe bij winst tegen random tegenstander
        elif is_winner(board, opponent[self.symbol]):
            reward = -1 #kent score -1 toe bij verlies tegen random tegenstander
        else:
            reward = 0 #kent score 0 toe bij gelijkspel tegen random tegenstander
        return reward

random.seed(1) #Begingetal patroon (seed) vastzetten voor betere vergelijking
 
my_agent = MyAgent(alpha=0.1, epsilon=0.1) #Hyperparameters van de slimme agent:
#Hoe snel pakt de agent nieuwe kennis op: alpha, 
#En hoe snel probeert de agent willekeurige acties probeert: epsilon
random_agent = RandomAgent() 
 
train_and_validate( #Functie voor trainen van agent 
    agent=my_agent,
    validation_agent=random_agent, #Tegenstander is de random agent
    iterations=50, #Aantal herhalingen trainproces
    trainings=500, #Aantal trainingen per herhaling
    validations=1000) #Aantal validatiespellen per herhaling

validation_agent = RandomAgent()

validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=1000) #Functie voor het valideren van de slimme agent

plot_validation(validation_result) #'tekenen' pie chart met resultaten