import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the Excel file using pandas
data = pd.read_excel("AirQualityUCI.xlsx")

# Preprocess data: Replace -200 with NaN and drop rows with NaN
data = data.replace(-200, np.nan).dropna()

# Select input features and target as specified
input_features = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
target = data.iloc[:, 5].values

# Normalize input features manually
def normalize_features(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    return (X - min_vals) / (max_vals - min_vals)

input_features = normalize_features(input_features)

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
        x = np.clip(x, -500, 500)  # Clip the values to avoid overflow
        return 1 / (1 + np.exp(-x))

    def calculate_mae(self, predictions, targets):
        return np.mean(np.abs(predictions - targets))

# PSO parameters
n_particles = 30
n_iterations = 100
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

        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (p_best_positions[i] - particles[i])
                             + c2 * r2 * (g_best_position - particles[i]))
            particles[i] += velocities[i]
    
    return g_best_position

# Splitting data into training and testing sets (10% for testing)
train_size = int(0.9 * len(input_features))
X_train, X_test = input_features[:train_size], input_features[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Train and evaluate the MLP with PSO
mlp_model = MLP(input_size=8, hidden_layers=[10, 10], output_size=1)
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
print(f"Mean Absolute Error (MAE) on test set: {mae}")

# Plotting the predicted vs actual values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Benzene Concentration", color="blue")
plt.plot(predictions, label="Predicted Benzene Concentration", color="red")
plt.xlabel("Sample")
plt.ylabel("Benzene Concentration")
plt.title("Actual vs Predicted Benzene Concentration")
plt.legend()
plt.show()
