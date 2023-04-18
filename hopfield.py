import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class HopfieldNetwork:
    def __init__(self, relaxation, beta, weights, tau, N, 
                 num_nodes_per_module, energy_func=None):
        self.relaxation = relaxation  # Number of relaxation steps
        self.beta = beta  # Inverse temperature parameter
        self.N = N  # Number of neurons
        self.weights = weights.astype('float64')  # Connection weights between neurons
        
        self.tau = tau  # Inter-reset duration
        
        self.energies = []  # List to store energy values
        
        self.relax_energies = []  # List to store the energy of initial relaxations
        
        # Add a list to store the inter-module energies
        self.inter_module_energies = []
        
        # set the number of nodes per module
        self.num_nodes_per_module = num_nodes_per_module
        
        # precompute the inter-module interaction terms
        self.inter_module_interaction = self.precompute_inter_module_interaction()

        # Set custom energy function or use default
        if energy_func:
            self.energy = energy_func
        else:
            self.energy = self.default_energy


    # Default energy function
    def default_energy(self, state):
        # Compute the energy of the given state using the Hopfield network energy function
        return -0.5 * np.dot(np.dot(state, self.weights), state)
    
    # Generate a new random state for each relaxation
    def generate_new_init_state(self):
        # Create a random initial state with -1 or 1 values for each neuron
        return np.random.choice([-1, 1], size=self.N)
    
    # Precompute inter-module interaction matrix
    def precompute_inter_module_interaction(self):
        if self.num_nodes_per_module == 0:
            return
        n = self.N // self.num_nodes_per_module
        inter_module_interaction = np.zeros((n, n))
        
        # Compute the interaction between different modules
        for i in range(self.N):
            for j in range(i + 1, self.N):
                module_i = i // self.num_nodes_per_module
                module_j = j // self.num_nodes_per_module
                
                if module_i != module_j:
                    inter_module_interaction[module_i, module_j] += self.weights[i, j]
                    inter_module_interaction[module_j, module_i] = inter_module_interaction[module_i, module_j]
                    
        return inter_module_interaction
    
    # Compute inter-module energy efficiently
    def inter_module_energy_fast(self, state):
        n = self.N // self.num_nodes_per_module
        avg_module_states = np.zeros(n)
        
        # Calculate the average state of each module
        for i in range(n):
            start_idx = i * self.num_nodes_per_module
            end_idx = start_idx + self.num_nodes_per_module
            avg_module_states[i] = np.mean(state[start_idx:end_idx])
        
        # Compute inter-module energy
        inter_module_energy = -0.5 * np.sum(self.inter_module_interaction * np.outer(avg_module_states, avg_module_states))
        return inter_module_energy
    
    # Update the state using the Heaviside step function
    def update_heavyside(self, state, i):
        activation = np.dot(self.weights[:, i], state)
        new_state = np.where(activation >= 0, 1, -1)
        return new_state

    # Update the state using the Boltzmann distribution
    def update_boltzmann(self, state, i):
        activation = np.dot(self.weights[:, i], state)
        probability = 1 / (1 + np.exp(-2 * self.beta * activation))
        new_state = np.where(np.random.rand() < probability, 1, -1)
        return new_state

    # Update the state deterministically using the Boltzmann distribution
    def deterministic_update_boltzmann(self, state, k):
        updated_state = np.copy(state)
        # Compute energy difference when flipping neuron k
        flipped_state = np.copy(state)
        flipped_state[k] = -flipped_state[k]
        energy_diff = self.energy(flipped_state) - self.energy(state)
        
        # Update the state of neuron k using the threshold function
        if energy_diff <= 0:
            updated_state[k] = flipped_state[k]
        
        return updated_state
    
    # train the network
    def train(self):
        # Iterate through all relaxations
        for i in range(self.relaxation):
            # Initialize the state for the current relaxation
            state = self.init_states[i]
            
            # Iterate through state updates
            for j in range(self.tau):
                # Update the index of the neuron to be updated (cycling through neurons)
                j = j % self.N
                k = self.update_order[j]
                
                # Update the state of neuron k using the heavyside update rule
                state[k] = self.update_heavyside(state, k)
                
                # Calculate the energy of the current state
                energy_i = self.energy(state)
                
                # Save the energies for the first 10 relaxations
                if i < 10:
                    self.relax_energies.append(energy_i)
                
            # Save the energy at relaxation i after self.tau steps without learning   
            print(f"Energy at relaxation {i+1} after {self.tau} steps without learning is {energy_i:.3f}")
            self.energies.append(energy_i)
        
        # Print the minimum energy found
        print(f"Minimum energy is {np.min(self.energies):.3f}")
        
    
    # Plot energy versus the number of state updates for the first n relaxations
    def plot_energy_vs_state_updates(self, n=5):
        for i in range(n):
            start_idx = i * self.tau
            end_idx = (i + 1) * self.tau
            energy_line = self.relax_energies[start_idx:end_idx]
            plt.plot(range(len(energy_line)), energy_line, label=f'Relaxation {i+1}')
            
        plt.xlabel('State updates')
        plt.ylabel('Energy')
        plt.title('Energy vs State Updates')
        plt.legend()
        plt.show()

    # Display the inter-module energy contributions for the first n relaxations
    def display_inter_module_energy_contributions(self, n=5):
        for i in range(n):
            start_idx = i * self.tau
            end_idx = (i + 1) * self.tau
            energy_line = self.inter_module_energies[start_idx:end_idx]
            plt.plot(range(len(energy_line)), energy_line, label=f'Relaxation {i+1}')
        
        plt.xlabel('State updates')
        plt.ylabel('Inter-module energy')
        plt.title('Inter-module Energy Contributions')
        plt.legend()
        plt.show()

    
    
    
