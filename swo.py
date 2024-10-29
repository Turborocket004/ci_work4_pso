import math
import random

# Load data manually from the provided text file
data = []
with open("/mnt/data/AirQualityUCI.txt", "r") as file:
    for line in file:
        values = line.strip().split("\t")
        # Replace missing values (-200) with NaN and filter out such rows
        if "-200" not in values:
            data.append([float(val) for val in values])

# Convert data to appropriate format (list of lists)
# Selecting columns 3, 6, 8, 10, 11, 12, 13, 14 for input and column 5 for output
input_features = [[row[3], row[6], row[8], row[10], row[11], row[12], row[13], row[14]] for row in data]
target = [row[5] for row in data]

# Split data manually into train and test sets (10% cross-validation)
def split_data(features, targets, test_ratio=0.1):
    total_size = len(features)
    test_size = int(total_size * test_ratio)
    indices = list(range(total_size))
    random.shuffle(indices)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = [features[i] for i in train_indices]
    y_train = [targets[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [targets[i] for i in test_indices]
    return X_train, y_train, X_test, y_test

# MLP class without libraries
class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            # Initialize weights and biases randomly
            self.layers.append({
                'weights': [[random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])] for _ in range(layer_sizes[i])],
                'bias': [random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])]
            })

    def forward(self, x):
        for layer in self.layers:
            x = self.sigmoid([sum(x[j] * layer['weights'][j][k] for j in range(len(x))) + layer['bias'][k] 
                              for k in range(len(layer['bias']))])
        return x[0]  # Output single value for regression

    def sigmoid(self, x):
        return [1 / (1 + math.exp(-val)) for val in x]

    def calculate_mae(self, predictions, targets):
        return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)

# PSO parameters
n_particles = 30
n_iterations = 100
c1, c2 = 1.5, 1.5
w = 0.5

# PSO function for optimizing MLP
def particle_swarm_optimization(mlp, X_train, y_train):
    n_weights = sum(len(layer['weights']) * len(layer['weights'][0]) + len(layer['bias']) for layer in mlp.layers)
    
    # Initialize particles and velocities
    particles = [[random.uniform(-1, 1) for _ in range(n_weights)] for _ in range(n_particles)]
    velocities = [[random.uniform(-1, 1) for _ in range(n_weights)] for _ in range(n_particles)]
    p_best_positions = particles.copy()
    p_best_scores = [float('inf')] * n_particles
    g_best_position = None
    g_best_score = float('inf')

    for iteration in range(n_iterations):
        for i in range(n_particles):
            mlp_flat = particles[i]
            index = 0
            for layer in mlp.layers:
                # Flatten weights and biases into the particle structure
                for j in range(len(layer['weights'])):
                    layer['weights'][j] = mlp_flat[index:index+len(layer['weights'][j])]
                    index += len(layer['weights'][j])
                layer['bias'] = mlp_flat[index:index+len(layer['bias'])]
                index += len(layer['bias'])
            
            # Evaluate particle performance
            predictions = [mlp.forward(x) for x in X_train]
            error = mlp.calculate_mae(predictions, y_train)

            # Update personal and global bests
            if error < p_best_scores[i]:
                p_best_scores[i] = error
                p_best_positions[i] = particles[i]
            if error < g_best_score:
                g_best_score = error
                g_best_position = particles[i]

        # Update velocity and position of each particle
        for i in range(n_particles):
            for j in range(n_weights):
                r1, r2 = random.random(), random.random()
                velocities[i][j] = (w * velocities[i][j]
                                    + c1 * r1 * (p_best_positions[i][j] - particles[i][j])
                                    + c2 * r2 * (g_best_position[j] - particles[i][j]))
                particles[i][j] += velocities[i][j]

    return g_best_position

# Train and evaluate the MLP with PSO
X_train, y_train, X_test, y_test = split_data(input_features, target, test_ratio=0.1)

# Create and optimize MLP model
mlp_model = MLP(input_size=8, hidden_layers=[10, 10], output_size=1)
best_weights = particle_swarm_optimization(mlp_model, X_train, y_train)

# Apply best weights to the MLP model
index = 0
for layer in mlp_model.layers:
    for j in range(len(layer['weights'])):
        layer['weights'][j] = best_weights[index:index+len(layer['weights'][j])]
        index += len(layer['weights'][j])
    layer['bias'] = best_weights[index:index+len(layer['bias'])]
    index += len(layer['bias'])

# Make predictions and evaluate MAE
predictions = [mlp_model.forward(x) for x in X_test]
mae = mlp_model.calculate_mae(predictions, y_test)
print(f"Mean Absolute Error (MAE) on test set: {mae}")
