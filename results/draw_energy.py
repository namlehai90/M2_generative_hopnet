import numpy as np
import matplotlib.pyplot as plt

def load_multiple_energies(seed_range):
    energies_list = []
    for seed in seed_range:
        energy_file = f"energy-{seed}.txt"
        energies = np.loadtxt(energy_file)
        energies_list.append(energies)
    return np.array(energies_list)

def plot_nonlearning_learning_energies_avg(seed_range):
    energies_array = load_multiple_energies(seed_range)
    avg_energies = energies_array.mean(axis=0)
    nl_relaxation = len(avg_energies) // 2
    total_relaxation = len(avg_energies)
    nonlearning_energies = avg_energies[:nl_relaxation]
    learning_energies = avg_energies[nl_relaxation:]

    plt.figure()
    plt.scatter(range(nl_relaxation), nonlearning_energies, label="rHN-0", marker='o', s=5)
    plt.scatter(range(nl_relaxation, total_relaxation), learning_energies, label="rHN-G", marker='x', s=5)
    
    # Add lines for the turning point and lowest non-learning energy
    plt.axvline(x=nl_relaxation - 0.5, color='red', linestyle='--')
    min_nonlearning_energy = np.min(nonlearning_energies)
    plt.axhline(y=min_nonlearning_energy, color='green', linestyle='-.')
    
    plt.xlabel("Relaxation")
    plt.ylabel("Energy")
    plt.title("Non-learning and learning energies")
    plt.legend()
    plt.savefig('energy_scatter.png')
    plt.show()

# Call the function with the seed range
plot_nonlearning_learning_energies_avg(range(1, 24))  # For seeds 1 to 10

def plot_energy_histograms_avg(seed_range):
    energies_array = load_multiple_energies(seed_range)
    nl_relaxation = energies_array.shape[1] // 2
    nonlearning_energies = energies_array[:, :nl_relaxation].flatten()
    learning_energies = energies_array[:, nl_relaxation:].flatten()

    plt.figure()
    plt.hist(nonlearning_energies, bins='auto', alpha=0.5, label='Before learning')
    plt.hist(learning_energies, bins='auto', alpha=0.5, label='After learning')
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Histogram of energies before and after learning")
    plt.legend()
    plt.savefig('histogram.png')
    plt.show()

# Call the function with the seed range
plot_energy_histograms_avg(range(0, 24))  # For seeds 1 to 10
