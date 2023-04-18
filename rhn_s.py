import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from hopfield import HopfieldNetwork

class rHN_S(HopfieldNetwork):
    """
    A class that implements rHN_S, a modified Hopfield network that updates the test part of the generate-and-test process
    by using a modified energy function accounting for learned weights.
    """
    def __init__(self, relaxation, beta, weights, tau, N, num_nodes_per_module, delta=0.0003):
        """
        Initialize the rHN_S with given parameters.
        
        :param relaxation: The number of relaxation iterations.
        :param beta: The inverse temperature parameter for the Boltzmann update.
        :param weights: The initial weight matrix.
        :param tau: The number of state updates within each relaxation iteration.
        :param N: The total number of state variables.
        :param num_nodes_per_module: The number of nodes per module in the modular constraint problem.
        :param delta: The learning rate for Hebbian learning (default=0.0003).
        """
        
        super().__init__(relaxation, beta, weights, tau, N, num_nodes_per_module)
        
        # learning rate
        self.delta = delta
    
    def gamma(self, x):
        """
        Apply the gamma function element-wise to a given matrix or scalar.
        :param x: A matrix or scalar value.
        :return: Element-wise maximum and minimum of x.
        """
        return np.maximum(-1, np.minimum(1, x))
    
    
    def hebbian_update1(self, state):
        """
        Update the weights of the network using the first version of the Hebbian learning rule.
        :param state: The current state of the network.
        """
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.weights[i, j] = self.gamma(self.weights[i, j] + (self.delta) * state[i] * state[j])
        self.weights -= np.diag(np.diag(self.weights))
        
    def hebbian_update(self, state):
        """
        Update the weights of the network using the second version of 
        the Hebbian learning rule.
        :param state: The current state of the network.
        """
        state_matrix = np.outer(state, state)
        delta_weights = self.delta * state_matrix
        
        # Update the weights using the gamma function
        self.weights += delta_weights
        self.weights = self.gamma(self.weights)

    def train(self):
        for i in range(self.relaxation):
            # Get relaxation state
            state = self.generate_new_init_state()
            
            # Perform state updates within each relaxation iteration
            for j in range(self.tau):
                # Randomly select a state variable to update
                k = np.random.randint(0, self.N)
                
                # Update state using deterministic Boltzmann update
                state = self.deterministic_update_boltzmann(state, k)
        
            # Apply Hebbian learning at the end of each relaxation step
            self.hebbian_update(state)
            
            # Compute and save inter-module energy at relaxation i
            inter_energy_i = self.inter_module_energy_fast(state)
            
            print(f"Inter-module Energy at relaxation {i+1} after {self.tau} steps with learning is {inter_energy_i:.3f}")
            self.energies.append(inter_energy_i)
            
        print(f"Minimum energy is {np.min(self.energies):.3f}")
        # Save energies into a file (uncomment if needed)
        # np.savetxt("learning_energy.txt", self.energies)

def run_rHN_S_model(seed=0):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    print("seed = ", seed)
    
    def create_mc_matrix(rc_matrix, k):
        n = rc_matrix.shape[0]
        N = k * n
        mc_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i, N):
                block_i = i // k
                block_j = j // k
                mc_matrix[i, j] = rc_matrix[block_i, block_j]
                mc_matrix[j, i] = mc_matrix[i, j]
                
        return mc_matrix

    # Define the RC matrix
    n = 20
    p = 0.01
    rc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                rc_matrix[i, j] = 1
            else:
                rc_matrix[i, j] = np.random.choice([-p, p])
                rc_matrix[j, i] = rc_matrix[i, j]
                
    # Define the MC matrix
    k = 10
    N = k * n
    mc_matrix = create_mc_matrix(rc_matrix, k)
    
    
    # initialise the weights
    init_weights = mc_matrix
    # init_weights = np.random.uniform(-1, 1, size=(N, N))
    # np.fill_diagonal(init_weights, 0)
    
    # Initialize the Hopfield network
    tau = 2000
    beta=1.0
    relaxation = 1000
    delta = 0.0015#0.00075
    hn = rHN_S(relaxation, beta, weights=init_weights, tau=tau, N=N, num_nodes_per_module=k, delta=delta)
    
    # Train the Hopfield network
    hn.train()
    
    # Save the energies to a file
    file_name  = "results/rhns/energy-" + str(seed) + ".txt"
    np.savetxt(file_name, hn.energies)
    
if __name__=="__main__":
    # one run
    # run_rHN_S_model()
    
    # 30 independent runs
    for seed in range(0, 30):
        run_rHN_S_model(seed)



