import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel("AirQualityUCI.xlsx")

# Preprocess data: Replace -200 with NaN, then drop rows with NaN values
data = data.replace(-200, np.nan).dropna()

# Select input features and target
input_features = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
target = data.iloc[:, 5].values

# Normalize input features manually
def normalize_features(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    return (X - min_vals) / (max_vals - min_vals)

input_features = normalize_features(input_features)

# Create shifted targets for multi-day prediction (5-day and 10-day ahead)
target_5day = np.roll(target, -5)[:-5]
target_10day = np.roll(target, -10)[:-10]

# Truncate input features to match shifted targets
input_features_5day = input_features[:-5]
input_features_10day = input_features[:-10]

# Define MLP class
class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            self.layers.append({
                'weights': np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1])),
                'bias': np.random.uniform(-limit, limit, layer_sizes[i+1])
            })

    def forward(self, x):
        for layer in self.layers:
            x = self.sigmoid(np.dot(x, layer['weights']) + layer['bias'])
        return x

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def calculate_mae(self, predictions, targets):
        return np.mean(np.abs(predictions - targets))

# PSO parameters
n_particles = 30
n_iterations = 10  # Reduced iterations for faster execution
c1, c2 = 1.5, 1.5
w = 0.5

# PSO function
def particle_swarm_optimization(mlp, X_train, y_train):
    n_weights = sum(layer['weights'].size + layer['bias'].size for layer in mlp.layers)
    particles = np.random.uniform(-1, 1, (n_particles, n_weights))
    velocities = np.random.uniform(-1, 1, (n_particles, n_weights))
    p_best_positions = particles.copy()
    p_best_scores = np.full(n_particles, float('inf'))
    g_best_position = None
    g_best_score = float('inf')

    for iteration in range(n_iterations):
        for i in range(n_particles):
            mlp_flat = particles[i]
            index = 0
            for layer in mlp.layers:
                layer_size = layer['weights'].size + layer['bias'].size
                layer_weights_bias = mlp_flat[index:index+layer_size]
                layer['weights'] = layer_weights_bias[:layer['weights'].size].reshape(layer['weights'].shape)
                layer['bias'] = layer_weights_bias[layer['weights'].size:]
                index += layer_size

            predictions = np.array([mlp.forward(x) for x in X_train]).flatten()
            error = mlp.calculate_mae(predictions, y_train)

            if error < p_best_scores[i]:
                p_best_scores[i] = error
                p_best_positions[i] = particles[i]
            if error < g_best_score:
                g_best_score = error
                g_best_position = particles[i]

        print(f"Iteration {iteration + 1}/{n_iterations}, Best Error: {g_best_score}")

        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (p_best_positions[i] - particles[i])
                             + c2 * r2 * (g_best_position - particles[i]))
            particles[i] += velocities[i]
    
    return g_best_position

# Experiment with single hidden layer configurations
hidden_layer_configs = [[10], [20], [30]]  # Single hidden layer configurations
results = {"5_day": [], "10_day": []}

for config in hidden_layer_configs:
    print(f"\nTesting hidden layer configuration: {config}")

    # Set up for 5-day and 10-day predictions with 10-fold CV
    for horizon, (X, y) in zip(["5_day", "10_day"], [(input_features_5day, target_5day), (input_features_10day, target_10day)]):
        n_folds = 10
        fold_size = len(X) // n_folds
        mae_scores = []

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for fold in range(n_folds):
            print(f"\nHorizon: {horizon}, Fold: {fold + 1}/{n_folds}")

            test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
            train_indices = np.concatenate((indices[:fold * fold_size], indices[(fold + 1) * fold_size:]))

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Train and evaluate the MLP with PSO
            mlp_model = MLP(input_size=8, hidden_layers=config, output_size=1)
            best_weights = particle_swarm_optimization(mlp_model, X_train, y_train)

            # Apply best weights to the model
            index = 0
            for layer in mlp_model.layers:
                layer_size = layer['weights'].size + layer['bias'].size
                layer_weights_bias = best_weights[index:index+layer_size]
                layer['weights'] = layer_weights_bias[:layer['weights'].size].reshape(layer['weights'].shape)
                layer['bias'] = layer_weights_bias[layer['weights'].size:]
                index += layer_size

            # Predict and calculate MAE for validation
            predictions = np.array([mlp_model.forward(x) for x in X_test]).flatten()
            mae = mlp_model.calculate_mae(predictions, y_test)
            mae_scores.append(mae)
            print(f"Fold {fold + 1} - Mean Absolute Error (MAE): {mae}")

        average_mae = np.mean(mae_scores)
        results[horizon].append((config, average_mae))
        print(f"{horizon} - Hidden layers {config} - Average MAE: {average_mae}")

# Plot the results for different configurations
plt.figure(figsize=(10, 5))
for horizon, result in results.items():
    x_labels = [str(config) for config, _ in result]
    mae_values = [mae for _, mae in result]
    plt.plot(x_labels, mae_values, marker='o', label=f'{horizon} prediction')
plt.xlabel("Hidden Layer Configuration")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE for Different Hidden Layer Configurations")
plt.legend()
plt.show()
