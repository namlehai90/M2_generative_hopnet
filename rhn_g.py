import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# from hopfield import HopfieldNetwork
from rhn_s import rHN_S

class rHN_G(rHN_S):
    """
    A class that implements rHN_G, a modified Hopfield network that updates the test part of the generate-and-test process
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
    
    # generative association
    def generative_update(self, state, k=0):
        updated_state = np.copy(state)

        # Compute the correlation matrix C = I + M
        C = np.identity(self.N) + self.weights

        # Choose a random neuron X
        X = np.random.randint(0, self.N)
        # X = state[k]

        # Flip the state of neuron X
        updated_state[X] = -state[X]
        
        # mean magnitude of weights in the matrix
        a = np.mean(np.abs(self.weights))

        # Calculate random threshold r uniformly in the range (a, 1]
        r = np.random.uniform(a, 1)
        
        # Update other neurons' states based on the learned correlations
        for j in range(self.N):
            if j != X:
                # Calculate the connection weight between neurons X and j, as well as the correlation
                m_Xj = self.weights[X, j]
                c_Xj = C[X, j]
                
                # Check if the correlation is greater than the random threshold r
                if abs(c_Xj) > r:
                    # Calculate the Heaviside function value
                    heaviside_value = -1 if c_Xj * updated_state[X] < 0 else 1
                    
                    # With probability abs(m_Xj), flip the state of neuron j
                    prob = np.random.uniform(0, 1)
                    if prob < abs(m_Xj):
                        updated_state[j] = heaviside_value

        # Energy difference check with discrete Boltzmann update
        energy_diff = self.energy(updated_state) - self.energy(state)
        
        # Accept or reject the updated state based on the energy difference and the Boltzmann distribution
        if energy_diff <= 0:
            return updated_state
        else:
            return state
        
    def train(self):
        for i in range(self.relaxation):
            # getting relaxation state
            # state = self.init_states[i]
            state = self.generate_new_init_state()
                
            # Perform tau state updates using the modified generative operator
            for j in range(self.tau):
                state = self.generative_update(state)
                    
            # Perform Hebbian learning at the end of each relaxation step
            self.hebbian_update(state)
            
            # Compute the inter-module energy at relaxation i   
            inter_energy_i = self.inter_module_energy_fast(state)
            
            print(f"Inter-module Energy at relaxation {i+1} after {self.tau} steps with learning is {inter_energy_i:.3f}")
            self.energies.append(inter_energy_i)

        # Print the minimum energy achieved during training
        print(f"Minimum energy is {np.min(self.energies):.3f}")
        # Save energies into a file
        # np.savetxt("generative_learning_energy.txt", self.energies)

        
def run_rHN_G_model(seed=0):
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
    delta = 0.0003#0.00075, 0.0015, 0.0003
    hn = rHN_G(relaxation, beta, weights=init_weights, tau=tau, N=N, num_nodes_per_module=k, delta=delta)
    
    # Train the Hopfield network
    hn.train()
    
    # save the energies into a file
    file_name = "results/rhng/energy-" + str(seed) + ".txt"
    np.savetxt(file_name, hn.energies)
    
if __name__=="__main__":
    # one run
    # run_rHN_G_model()
    
    # 30 independent runs
    for seed in range(0, 30):
        run_rHN_G_model(seed)
