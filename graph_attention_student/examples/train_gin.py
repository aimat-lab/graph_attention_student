#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "torch-geometric>=2.5",
#   "lightning>=2.0",
#   "rdkit",
#   "numpy<2.0",
# ]
# ///
"""Minimal Example GIN Regressor"""
import torch
import torch.nn as nn
import numpy as np
import lightning as L
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# =============================================================================
# 1. DATASET
# =============================================================================
# A small hardcoded dataset of (SMILES, cLogP) pairs.
# In practice, this would be replaced with a CSV file containing thousands of molecules.
DATASET = [
    ("C",                      -0.59),
    ("CC",                      0.89),
    ("CCC",                     1.44),
    ("CCCC",                    1.99),
    ("CCCCC",                   2.54),
    ("CCCCCC",                  3.09),
    ("CCCCCCCC",                4.19),
    ("CC(C)C",                  2.13),
    ("CO",                     -0.63),
    ("CCO",                    -0.31),
    ("CCCO",                    0.25),
    ("CCCCO",                   0.88),
    ("C(=O)O",                 -0.90),
    ("CC(=O)O",                -0.17),
    ("CC=O",                   -0.34),
    ("CCC=O",                   0.59),
    ("CCN",                    -0.13),
    ("c1ccccc1",                1.56),
    ("c1ccc(C)cc1",             2.73),
    ("c1ccc(CC)cc1",            3.15),
    ("c1ccc(O)cc1",             1.46),
    ("c1ccc(N)cc1",             0.90),
    ("c1ccncc1",                0.65),
    ("c1ccc2ccccc2c1",          3.30),
    ("OC(=O)c1ccccc1",          1.87),
    ("CC(=O)Nc1ccccc1",         1.16),
]


# =============================================================================
# 2. FEATURIZATION — Converting SMILES to Graphs
# =============================================================================
# To feed molecules into a GNN, we need to represent them as graphs:
#   - Nodes = atoms, with feature vectors encoding chemical properties
#   - Edges = bonds, defined by the molecular connectivity
#
# We use RDKit to parse SMILES and extract atom/bond information.

# Atomic numbers we one-hot encode. Covers most of organic chemistry.
# Any atom not in this list gets the "other" category.
ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35]  # C, N, O, F, P, S, Cl, Br


def one_hot(value: int, choices: list) -> list:
    """
    Create a one-hot encoding vector.

    Example: one_hot(7, [6, 7, 8]) -> [0, 1, 0, 0]
    The last position is the "other" category for unknown values.
    """
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1  # "other" category
    return encoding


def atom_features(atom) -> list:
    """
    Compute a numerical feature vector for a single atom.

    Features used (23 total):
        - Atomic number, one-hot encoded (9 dims: 8 elements + "other")
        - Degree (number of bonds), one-hot (6 dims: 0-4 + "other")
        - Formal charge (1 dim: integer value)
        - Number of attached hydrogens, one-hot (6 dims: 0-4 + "other")
        - Is aromatic (1 dim: binary)
    """
    features = []

    # What element is this atom? (e.g., carbon=6, nitrogen=7, oxygen=8)
    features += one_hot(atom.GetAtomicNum(), ATOM_TYPES)

    # How many bonds does this atom have? (e.g., carbon typically has 4)
    features += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4])

    # Formal charge (e.g., 0 for most atoms, +1 for ammonium nitrogen)
    features.append(atom.GetFormalCharge())

    # How many hydrogens are attached? (implicit + explicit)
    features += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    # Is this atom part of an aromatic ring? (e.g., carbons in benzene)
    features.append(1 if atom.GetIsAromatic() else 0)

    return features


# Total number of features per atom (must match the model's input dimension)
# Each one_hot() adds len(choices)+1 dims (the +1 is the "other" category)
NUM_ATOM_FEATURES = 9 + 6 + 1 + 6 + 1  # = 23


def smiles_to_data(smiles: str, target: float) -> Data:
    """
    Convert a SMILES string and its target value into a PyG Data object.

    Steps:
        1. Parse SMILES with RDKit to get a molecule object
        2. Extract atom features for each atom -> node feature matrix
        3. Extract bond connectivity -> edge index (in both directions)
        4. Attach the target value as a graph-level label

    Returns:
        A torch_geometric.data.Data object with:
            - x:          Node feature matrix,  shape (num_atoms, 23)
            - edge_index: Edge connectivity,     shape (2, num_edges*2)
            - y:          Target value,          shape (1,)
    """
    from rdkit import Chem

    # Parse the SMILES string into an RDKit molecule object.
    # RDKit handles valence checking, aromaticity perception, etc.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    # --- Node features ---
    # Build a feature vector for every atom in the molecule.
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    # Convert to a float tensor: shape (num_atoms, NUM_ATOM_FEATURES)
    x = torch.tensor(x, dtype=torch.float)

    # --- Edge index ---
    # In PyG, edges are stored as a (2, num_edges) tensor where:
    #   edge_index[0] = source nodes
    #   edge_index[1] = destination nodes
    #
    # Molecular bonds are undirected, so we add edges in BOTH directions:
    #   bond A-B  ->  edges (A->B) and (B->A)
    # This ensures information flows both ways during message passing.
    edge_src, edge_dst = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions
        edge_src += [i, j]
        edge_dst += [j, i]

    # Convert to a long tensor: shape (2, num_edges*2)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # --- Target ---
    # Graph-level regression target (cLogP). Shape (1,) for batching.
    y = torch.tensor([target], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


# =============================================================================
# 3. GIN MODEL
# =============================================================================
# We define a GIN regression model as a PyTorch Lightning module.
# Architecture:
#   Input (23 atom features)
#     -> GINConv layer 1 (23 -> 64)
#     -> GINConv layer 2 (64 -> 64)
#     -> GINConv layer 3 (64 -> 64)
#     -> Global Add Pooling (sum all node embeddings -> single graph vector)
#     -> MLP (64 -> 32 -> 1)
#     -> Predicted cLogP


class GINRegressor(L.LightningModule):
    """
    A Graph Isomorphism Network for graph-level regression.

    GINConv works by:
        1. Summing the features of all neighboring nodes
        2. Adding the node's own features (scaled by a learnable epsilon)
        3. Passing the result through an MLP

    This is repeated for each GIN layer, building up richer node
    representations that capture larger neighborhoods of the graph.
    Finally, all node representations are summed (global add pooling)
    to produce a single fixed-size vector for the entire graph.
    """

    def __init__(
        self,
        node_dim: int = NUM_ATOM_FEATURES,
        hidden_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        # Save hyperparameters so Lightning can log & restore them
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # --- GIN Layers ---
        # Each GINConv wraps an MLP (multi-layer perceptron).
        # The MLP processes the aggregated neighborhood features.
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # First layer: input dim = atom features; subsequent: hidden_dim
            in_dim = node_dim if i == 0 else hidden_dim

            # The MLP inside GINConv: 2-layer with ReLU activation
            # This is applied AFTER neighborhood aggregation
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # GINConv with trainable epsilon (train_eps=True)
            # epsilon controls how much weight the node's own features get
            # relative to its neighbors' features
            self.gin_layers.append(GINConv(nn=mlp, train_eps=True))

            # Batch normalization stabilizes training by normalizing
            # activations across the batch
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # --- Readout MLP ---
        # After GIN layers, we pool all node embeddings into one graph vector
        # and pass it through a final MLP to predict the target property.
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),     # 64 -> 32
            nn.ReLU(),
            nn.Dropout(0.1),                             # light regularization
            nn.Linear(hidden_dim // 2, 1),               # 32 -> 1 (regression)
        )

    def forward(self, data):
        """
        Forward pass: graph -> predicted cLogP.

        Args:
            data: A PyG Batch object containing:
                - x:          Node features        (total_nodes, 23)
                - edge_index: Edge connectivity     (2, total_edges)
                - batch:      Maps each node to its graph in the batch

        Returns:
            Predicted cLogP values, shape (batch_size, 1)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through each GIN layer with batch norm and ReLU
        for gin_layer, bn in zip(self.gin_layers, self.batch_norms):
            x = gin_layer(x, edge_index)    # message passing + MLP
            x = bn(x)                        # normalize activations
            x = torch.relu(x)               # non-linearity

        # Global add pooling: sum all node embeddings per graph.
        # If the batch contains 4 graphs, this produces a (4, hidden_dim) tensor.
        # "Add" pooling is the standard choice for GIN because the GIN paper
        # proves that sum aggregation is maximally expressive for distinguishing
        # different graph structures.
        graph_embedding = global_add_pool(x, batch)

        # Final MLP maps graph embedding -> scalar prediction
        return self.readout(graph_embedding)

    def training_step(self, batch, batch_idx):
        """Compute MSE loss on a training batch."""
        y_hat = self.forward(batch)             # predicted values
        loss = nn.functional.mse_loss(y_hat, batch.y.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute MSE loss on a validation batch."""
        y_hat = self.forward(batch)
        loss = nn.functional.mse_loss(y_hat, batch.y.unsqueeze(1))
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        """Use Adam optimizer with the specified learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# =============================================================================
# 4. MAIN — Data Preparation, Training, and Evaluation
# =============================================================================

def main():
    # Suppress RDKit warnings (it's quite chatty about implicit Hs, etc.)
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    # --- Convert all SMILES to PyG Data objects ---
    print("Converting SMILES to molecular graphs...")
    dataset = []
    for smiles, target in DATASET:
        data = smiles_to_data(smiles, target)
        dataset.append(data)

    print(f"  Total molecules: {len(dataset)}")
    print(f"  Atom features per node: {NUM_ATOM_FEATURES}")
    print(f"  Example — {DATASET[0][0]}: {dataset[0].x.shape[0]} atoms, "
          f"{dataset[0].edge_index.shape[1]} directed edges")

    # --- Train/Validation Split ---
    # Shuffle the dataset and split 80% train / 20% validation.
    # We use a fixed random seed for reproducibility.
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(dataset))
    split = int(0.8 * len(dataset))

    train_data = [dataset[i] for i in indices[:split]]
    val_data = [dataset[i] for i in indices[split:]]
    print(f"  Train: {len(train_data)}, Validation: {len(val_data)}")

    # --- DataLoaders ---
    # PyG's DataLoader automatically batches graphs by:
    #   1. Concatenating all node features into one big tensor
    #   2. Adjusting edge indices to account for the offset
    #   3. Creating a `batch` vector that maps each node to its graph
    # This is much more efficient than padding all graphs to the same size.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    # --- Initialize Model ---
    model = GINRegressor(
        node_dim=NUM_ATOM_FEATURES,     # 23 input features per atom
        hidden_dim=64,                   # size of hidden representations
        num_layers=3,                    # 3 GIN message-passing layers
        learning_rate=1e-3,              # Adam learning rate
    )

    # Print a summary of the model architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} trainable parameters")

    # --- Train with PyTorch Lightning ---
    # The Trainer handles the training loop, logging, checkpointing, etc.
    trainer = L.Trainer(
        max_epochs=200,                  # train for 200 epochs
        log_every_n_steps=1,             # log metrics every step
        enable_checkpointing=False,      # skip saving checkpoints (this is a demo)
        enable_model_summary=False,      # we printed our own summary above
    )

    print("\nTraining...")
    trainer.fit(model, train_loader, val_loader)

    # --- Evaluate: Print Predictions vs Ground Truth ---
    print("\n" + "=" * 60)
    print("PREDICTIONS vs GROUND TRUTH (full dataset)")
    print("=" * 60)
    print(f"{'SMILES':<30} {'True cLogP':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 60)

    # Switch model to evaluation mode (disables dropout, etc.)
    model.eval()
    with torch.no_grad():
        for smiles, true_value in DATASET:
            # Convert single molecule and predict
            data = smiles_to_data(smiles, true_value)
            # Create a minimal batch (batch vector = all zeros = single graph)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            pred = model(data).item()
            error = pred - true_value
            print(f"{smiles:<30} {true_value:>10.2f} {pred:>10.2f} {error:>+10.2f}")

    # --- Compute overall metrics ---
    all_true, all_pred = [], []
    with torch.no_grad():
        for smiles, true_value in DATASET:
            data = smiles_to_data(smiles, true_value)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            pred = model(data).item()
            all_true.append(true_value)
            all_pred.append(pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    mae = np.mean(np.abs(all_true - all_pred))
    rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
    # R² = 1 - (sum of squared residuals) / (total variance)
    # R² = 1.0 means perfect prediction, 0.0 means no better than predicting the mean
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print("-" * 60)
    print(f"{'MAE (Mean Absolute Error)':<40} {mae:>10.3f}")
    print(f"{'RMSE (Root Mean Squared Error)':<40} {rmse:>10.3f}")
    print(f"{'R² (Coefficient of Determination)':<40} {r2:>10.3f}")


if __name__ == "__main__":
    main()
