import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from hopfield import HopfieldNetwork

class rHN_0(HopfieldNetwork):
    def __init__(self, relaxation, beta, weights, tau, N, num_nodes_per_module):
        # Call the parent class constructor
        super().__init__(relaxation, beta, weights, tau, N, num_nodes_per_module)
        
    def train(self):
        for i in range(self.relaxation):
            # Generate a new initial state for each relaxation
            state = self.generate_new_init_state()
                
            for j in range(self.tau):
                # Choose a random neuron to update
                k = np.random.randint(0, self.N)
            
                # Update the state using deterministic Boltzmann update
                state = self.deterministic_update_boltzmann(state, k)
                    
                # Calculate energy
                energy_i = self.energy(state)
                
                # Save energies for some initial relaxations
                if i < 5:
                    self.relax_energies.append(energy_i)
                    
                    # Calculate inter-module energy and store it
                    if self.num_nodes_per_module > 0:
                        inter_module_energy_i = self.inter_module_energy_fast(state)
                        self.inter_module_energies.append(inter_module_energy_i)
                    
            # Save inter-module energy at relaxation i
            inter_energy_i = self.inter_module_energy_fast(state)
            print(f"Inter-module Energy at relaxation {i+1} after {self.tau} steps without learning is {inter_energy_i:.3f}")
            self.energies.append(inter_energy_i)
        print(f"Minimum energy is {np.min(self.energies):.3f}")
    
        # Save energies into a file
        np.savetxt("nonlearning_energy.txt", self.energies)

# run rHN-0
def run_rHN_0_model(seed=0):
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

    # Initialize the weight matrix
    init_weights = mc_matrix
    
    # Initialize the rHN_0 Hopfield network
    tau = 2000
    beta = 1.0
    relaxation = 1000
    hn = rHN_0(relaxation, beta, weights=init_weights, tau=tau, N=N, num_nodes_per_module=k)
    
    # Train the Hopfield network
    hn.train()
    
    # Save the network's energies to a file
    file_name = "results/rhn0/energy-" + str(seed) + ".txt"
    np.savetxt(file_name, hn.energies)

    # Optionally, plot the energy and inter-module energy contributions
    # hn.plot_energy_vs_state_updates()
    # hn.display_inter_module_energy_contributions()

if __name__=="__main__":
    # one run
    # run_rHN_0_model()
    
    # 30 independent runs
    for seed in range(0, 30):
        run_rHN_0_model(seed)