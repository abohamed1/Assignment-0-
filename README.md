# Assignment-0-

import math
import random
import numpy as np

#Simulated_annealing
def simulated_annealing(cost_function, initial_state, temperature, cooling_rate, iterations):
    current_state = initial_state
    current_cost = cost_function(initial_state)
    
    for i in range(iterations):
        temperature *= cooling_rate
        next_state = generate_neighbor(current_state)
        next_cost = cost_function(next_state)
        
        if next_cost < current_cost or random.random() < math.exp((current_cost - next_cost) / temperature):
            current_state = next_state
            current_cost = next_cost
    
    return current_state, current_cost

def generate_neighbor(state):
    # Example: randomly change one element in the state
    neighbor = state.copy()
    index = random.randint(0, len(state) - 1)
    neighbor[index] = random.uniform(-10, 10)
    return neighbor

# Example cost function: minimize the sum of squares
def cost_function(state):
    return sum(x ** 2 for x in state)

initial_state = [random.uniform(-10, 10) for _ in range(5)]
temperature = 100.0
cooling_rate = 0.95
iterations = 1000

result_state, result_cost = simulated_annealing(cost_function, initial_state, temperature, cooling_rate, iterations)
print("Simulated Annealing:")
print("Result State:", result_state)
print("Result Cost:", result_cost)


#Hill_Climbing
def hill_climbing(cost_function, initial_state, step_size, iterations):
    current_state = initial_state
    current_cost = cost_function(initial_state)
    
    for _ in range(iterations):
        next_state = generate_neighbor(current_state, step_size)
        next_cost = cost_function(next_state)
        
        if next_cost < current_cost:
            current_state = next_state
            current_cost = next_cost
        else:
            break
    
    return current_state, current_cost

def generate_neighbor(state, step_size):
    # Example: randomly change one element in the state
    index = random.randint(0, len(state) - 1)
    neighbor = state.copy()
    neighbor[index] += random.uniform(-step_size, step_size)
    return neighbor

# Example cost function: minimize the sum of squares
def cost_function(state):
    return sum(x ** 2 for x in state)

initial_state = [random.uniform(-10, 10) for _ in range(5)]
step_size = 0.1
iterations = 1000

result_state, result_cost = hill_climbing(cost_function, initial_state, step_size, iterations)
print("Hill Climbing:")
print("Result State:", result_state)
print("Result Cost:", result_cost)


#Gradient_descent
def gradient_descent(gradient_function, initial_guess, learning_rate, iterations):
    current_guess = initial_guess
    
    for _ in range(iterations):
        gradient = gradient_function(current_guess)
        current_guess = current_guess - learning_rate * gradient
    
    return current_guess

# Example cost function: minimize the sum of squares
def cost_function(state):
    return np.sum(np.square(state))

# Example gradient function: derivative of the cost function
def gradient_function(state):
    return 2 * state

initial_guess = np.array([random.uniform(-10, 10) for _ in range(5)])
learning_rate = 0.1
iterations = 1000

result_state = gradient_descent(gradient_function, initial_guess, learning_rate, iterations)
result_cost = cost_function(result_state)

print("Gradient Descent:")
print("Result State:", result_state)
print("Result Cost:", result_cost)
