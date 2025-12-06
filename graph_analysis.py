import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from sklearn.metrics import roc_auc_score, mean_squared_error
import gc
import argparse
import sys
import os

# ==========================================
# 1. Core Vectorized Graph Algorithms
# ==========================================

def calculate_adamic_adar_vectorized(adj_matrix, u_nodes, v_nodes, batch_size=10000):
    """
    Calculates Adamic-Adar index efficiently using sparse matrix operations.
    
    Args:
        adj_matrix (scipy.sparse.csr_matrix): The graph adjacency matrix. Recommend symmetrizing it first: A + A.T
        u_nodes (array-like): Source node indices.
        v_nodes (array-like): Target node indices.
        batch_size (int): Batch size to manage RAM.
        
    Returns:
        np.array: Adamic-Adar scores for each pair.
    """
    n_pairs = len(u_nodes)
    scores = np.zeros(n_pairs)
    
    # 1. Pre-calculate the '1/log(degree)' weights
    # We use the undirected degree (sum of row + col usually, or just row if symmetric)
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Handle division by zero or log(1)=0.
    # Nodes with degree <= 1 get weight 0 (they don't contribute to connectivity)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_log_degrees = 1.0 / np.log(degrees)
        inv_log_degrees[degrees <= 1] = 0.0
        inv_log_degrees[np.isinf(inv_log_degrees)] = 0.0
        
    # Create a diagonal matrix of weights
    # shape: (n_nodes, n_nodes)
    D_diag = sparse.diags(inv_log_degrees)
    
    # 2. Create the Weighted Adjacency Matrix
    # A_weighted[i, j] = 1 / log(degree(j)) if connected
    # We multiply the adjacency matrix by the diagonal weight matrix
    # Note: To match AA definition, weights apply to the shared neighbor 'w'
    adj_weighted = adj_matrix.dot(D_diag)
    
    # 3. Process in batches to save RAM (Vectorized Row-wise Dot Product)
    for i in range(0, n_pairs, batch_size):
        end = min(i + batch_size, n_pairs)
        idx_u = u_nodes[i:end]
        idx_v = v_nodes[i:end]
        
        # Slice the matrices: Get rows for u and rows for v
        # shape: (batch_size, n_nodes)
        rows_u = adj_weighted[idx_u]
        rows_v = adj_matrix[idx_v]
        
        # The 'Magic' Step:
        # Instead of dot product (which results in batch x batch),
        # we do element-wise multiplication and sum across the row.
        # This calculates exactly sum(weight * 1) for shared indices.
        batch_scores = rows_u.multiply(rows_v).sum(axis=1)
        
        # The result is a numpy matrix (n_batch, 1), flatten to array
        scores[i:end] = np.array(batch_scores).flatten()
        
    return scores

# ==========================================
# 2. Feature Engineering
# ==========================================

def get_smoothed_avg_weight(node_ids, weight_sum, degrees, global_avg_weight, C=5):
    """
    Bayesian Smoothing for average weights.
    Formula: (Sum_Weights + C * Global_Avg) / (Degree + C)
    """
    local_sums = weight_sum[node_ids]
    local_degrees = degrees[node_ids]
    
    smoothed_avg = (local_sums + C * global_avg_weight) / (local_degrees + C)
    return smoothed_avg

def build_features(u_nodes, v_nodes, graph_csr, node_props, degrees, weight_stats=None):
    """
    Constructs the feature matrix for a list of node pairs.
    
    Args:
        u_nodes, v_nodes: Arrays of node IDs.
        graph_csr: Sparse adjacency matrix (symmetric/undirected for topology).
        node_props: DataFrame where index = node_ID.
                   Columns are multiple features.
        degrees: Pre-calculated degrees of nodes.
        weight_stats (dict, optional): Contains 'sum_out', 'sum_in', 'global_avg' for weight features.
        
    Returns:
        X: DataFrame or numpy array of features.
    """
    # --- Topological Features ---
    
    # 1. Preferential Attachment (Degree Product)
    feat_pref_attach = degrees[u_nodes] * degrees[v_nodes]
    
    # 2. Adamic-Adar (Vectorized)
    # Assumes graph_csr is symmetrized for this calculation as recommended
    feat_adamic = calculate_adamic_adar_vectorized(graph_csr, u_nodes, v_nodes)
    
    feature_dict = {
        'pref_attach': feat_pref_attach,
        'adamic_adar': feat_adamic,
    }
    
    # --- Node Property Features ---
    if isinstance(node_props, pd.DataFrame):
        # Assume node_props is indexed by node_id 0..N
        # We need to extract features for u and v
        
        # For each column in node_props, create u_feature, v_feature, and interaction
        for col in node_props.columns:
            if col == 'node_id': continue
            
            # Extract raw values
            u_vals = node_props.loc[u_nodes, col].values
            v_vals = node_props.loc[v_nodes, col].values
            
            feature_dict[f'u_{col}'] = u_vals
            feature_dict[f'v_{col}'] = v_vals
            
            # Interaction: If numeric, maybe diff? If categorical, maybe match?
            # For now, simple equality check (works for both)
            # Convert to numeric for LightGBM if needed, but LGBM handles categorical integers well
            feature_dict[f'same_{col}'] = (u_vals == v_vals).astype(int)
    else:
        print("Error: node_props is not a DataFrame. Feature extraction requires a DataFrame.")
        sys.exit(1)

    # --- Weight/Activity Features (for Stage 2 or enhanced Stage 1) ---
    if weight_stats:
        # Bayesian Smoothed Average Weights
        # Avg Weight of U (Outgoing/General activity)
        feat_avg_w_u = get_smoothed_avg_weight(
            u_nodes, weight_stats['sum_out'], degrees, weight_stats['global_avg']
        )
        
        # Avg Weight of V (Incoming/Popularity intensity)
        feat_avg_w_v = get_smoothed_avg_weight(
            v_nodes, weight_stats['sum_in'], degrees, weight_stats['global_avg']
        )
        
        # Resource Allocation approximation (1/Degree(u))
        # Avoid div by zero
        safe_deg_u = degrees[u_nodes].astype(float)
        safe_deg_u[safe_deg_u == 0] = 1.0
        feat_res_alloc_u = 1.0 / safe_deg_u
        
        feature_dict['avg_weight_u'] = feat_avg_w_u
        feature_dict['avg_weight_v'] = feat_avg_w_v
        feature_dict['res_alloc_u'] = feat_res_alloc_u

    return pd.DataFrame(feature_dict)

def generate_negatives(num_neg, num_nodes, existing_edge_set):
    """
    Generates negative samples (edges that do not exist) efficiently.
    
    Args:
        num_neg: Number of negative samples required.
        num_nodes: Total number of nodes.
        existing_edge_set: Set of existing edges encoded as integers (u * num_nodes + v).
        
    Returns:
        neg_u, neg_v: Arrays of negative sample edges.
    """
    neg_u_list = []
    neg_v_list = []
    current_count = 0
    
    while current_count < num_neg:
        needed = num_neg - current_count
        # Oversample slightly to handle collisions and self-loops
        batch_size = int(needed * 1.2) + 100
        
        u_cand = np.random.randint(0, num_nodes, batch_size)
        v_cand = np.random.randint(0, num_nodes, batch_size)
        
        # Filter self-loops
        mask_loops = u_cand != v_cand
        u_cand = u_cand[mask_loops]
        v_cand = v_cand[mask_loops]
        
        # Filter existing edges
        cand_ids = u_cand.astype(np.int64) * num_nodes + v_cand
        
        # Check against set (this list comprehension is reasonably fast for <10M items)
        mask_not_existing = np.array([cid not in existing_edge_set for cid in cand_ids])
        
        u_valid = u_cand[mask_not_existing]
        v_valid = v_cand[mask_not_existing]
        
        neg_u_list.append(u_valid)
        neg_v_list.append(v_valid)
        
        current_count += len(u_valid)
        
    # Concatenate and slice to exact number
    neg_u = np.concatenate(neg_u_list)[:num_neg]
    neg_v = np.concatenate(neg_v_list)[:num_neg]
    
    return neg_u, neg_v

# ==========================================
# 3. Main Pipeline
# ==========================================

def predict_all_unknown_edges(num_nodes, existing_edge_set, graph_csr_sym, node_props, degrees, weight_stats, model_existence, model_weight, output_file="predicted_edges.csv", threshold=0.5):
    """
    Iterates through ALL possible non-edges in the graph and predicts their probability and weight.
    Writes results directly to a CSV to avoid memory issues.
    """
    print(f"\n--- Starting Full Graph Prediction (Unknown Edges Only) ---")
    print(f"Total possible pairs: {num_nodes * num_nodes}")
    print(f"Targeting output file: {output_file}")
    
    # Open file and write header
    with open(output_file, 'w') as f:
        # Determine header from a dummy feature construction to get column names correct
        dummy_u = np.array([0])
        dummy_v = np.array([1])
        # We output u, v, prob, pred_weight. We don't output features to save space.
        f.write("u,v,prob_exists,pred_weight\n")
    
    # Parameters for batching
    # We iterate over source nodes in chunks
    source_batch_size = 100  # Small batch of source nodes
    
    all_nodes = np.arange(num_nodes)
    
    # Stats
    total_predictions = 0
    edges_found = 0
    
    # Iterate over all source nodes u
    for u_start in range(0, num_nodes, source_batch_size):
        u_end = min(u_start + source_batch_size, num_nodes)
        u_batch = np.arange(u_start, u_end)
        
        # For each source node in this small batch, we want to check ALL target nodes
        # But to be efficient, we can create a grid
        # Grid size: source_batch_size * num_nodes (e.g., 100 * 60,000 = 6M pairs)
        # This fits in memory for feature construction
        
        # Repeat u for each v
        u_grid = np.repeat(u_batch, num_nodes)
        # Tile v for each u
        v_grid = np.tile(all_nodes, len(u_batch))
        
        # --- Filtering ---
        # 1. Filter self-loops
        mask_loops = u_grid != v_grid
        u_cand = u_grid[mask_loops]
        v_cand = v_grid[mask_loops]
        
        # 2. Filter existing edges
        cand_ids = u_cand.astype(np.int64) * num_nodes + v_cand
        
        # Efficient boolean mask using set
        # Note: list comp is fast enough for checking existence
        mask_unknown = np.array([cid not in existing_edge_set for cid in cand_ids])
        
        u_final = u_cand[mask_unknown]
        v_final = v_cand[mask_unknown]
        
        if len(u_final) == 0:
            continue
            
        # --- Prediction ---
        # Build features
        X_batch = build_features(u_final, v_final, graph_csr_sym, node_props, degrees, weight_stats)
        
        # Predict Existence
        probs = model_existence.predict_proba(X_batch)[:, 1]
        
        # Filter by threshold to save disk space and time
        mask_likely = probs > threshold
        
        if np.sum(mask_likely) > 0:
            u_likely = u_final[mask_likely]
            v_likely = v_final[mask_likely]
            probs_likely = probs[mask_likely]
            X_likely = X_batch[mask_likely]
            
            # Predict Weight for likely edges
            weights_log = model_weight.predict(X_likely)
            weights_real = np.expm1(weights_log)
            
            # Write to file
            # Use pandas for easy CSV formatting of chunk
            df_results = pd.DataFrame({
                'u': u_likely,
                'v': v_likely,
                'prob_exists': probs_likely,
                'pred_weight': weights_real
            })
            
            df_results.to_csv(output_file, mode='a', header=False, index=False)
            edges_found += len(df_results)
            
        total_predictions += len(u_final)
        
        if u_end % 1000 == 0:
            print(f"Processed nodes up to {u_end}/{num_nodes}. Found {edges_found} potential edges so far...")
            gc.collect() # Force garbage collection
            
    print(f"\nFinished. Scanned {total_predictions} unknown pairs.")
    print(f"Found {edges_found} edges above probability threshold {threshold}.")
    print(f"Results saved to {output_file}")

def run_pipeline(input_csv=None, features_csv=None, output_csv="predicted_edges.csv", threshold=0.5):
    num_nodes = 60000 # Default
    
    # -------------------------------------------------------
    # A. Data Loading
    # -------------------------------------------------------
    df_edges = None
    if input_csv:
        print(f"--- Loading Data from {input_csv} ---")
        try:
            # Assume standard columns or first 3 columns are u, v, weight
            df_edges = pd.read_csv(input_csv)
            
            # Standardize column names
            if len(df_edges.columns) >= 3:
                df_edges.columns = ['u', 'v', 'weight'] + list(df_edges.columns[3:])
            else:
                print("Error: Input CSV must have at least 3 columns (u, v, weight)")
                sys.exit(1)

            # Remap node IDs check
            max_id = max(df_edges['u'].max(), df_edges['v'].max())
            if max_id >= 60000:
                num_nodes = max_id + 1
                print(f"Detected {num_nodes} nodes from input file.")
                
        except Exception as e:
            print(f"Failed to read CSV: {e}")
            sys.exit(1)
    else:
        print("Error: Input CSV file (--input) is required. Synthetic generation has been disabled.")
        sys.exit(1)

    # Clean up duplicates
    df_edges = df_edges.drop_duplicates(subset=['u', 'v'])
    print(f"Final edges count: {len(df_edges)}")
    
    # -------------------------------------------------------
    # A.2 Feature Loading
    # -------------------------------------------------------
    node_props = None
    cat_features = []
    
    if features_csv:
        print(f"--- Loading Node Features from {features_csv} ---")
        try:
            df_features = pd.read_csv(features_csv)
            # Expecting: node_id, feat1, feat2...
            # Ensure node_id matches our node indices
            if 'node_id' not in df_features.columns:
                 # assume first column is node_id if not explicit
                 df_features.rename(columns={df_features.columns[0]: 'node_id'}, inplace=True)
            
            # Index by node_id for fast lookups
            # We reindex to ensure we have a row for every node 0..num_nodes-1
            # Fill missing nodes with 0 or -1 (handling as categorical or default)
            df_features.set_index('node_id', inplace=True)
            
            # Reindex to full range 0 to num_nodes-1
            # This ensures df_features.loc[0] works even if 0 was missing in file
            df_features = df_features.reindex(range(num_nodes), fill_value=0)
            
            node_props = df_features
            print(f"Loaded features for {len(node_props)} nodes. Columns: {list(node_props.columns)}")
            
            # Identify categorical columns for LightGBM
            # Construct feature names: u_col, v_col, same_col
            for col in node_props.columns:
                cat_features.extend([f'u_{col}', f'v_{col}', f'same_{col}'])
            
        except Exception as e:
            print(f"Failed to read Features CSV: {e}")
            sys.exit(1)
    else:
        print("Error: Node features CSV file (--features) is required. Synthetic generation has been disabled.")
        sys.exit(1)

    # --- Create Set of Existing Edges for Fast Lookup ---
    print("Indexing existing edges...")
    existing_edge_set = set(df_edges['u'].values.astype(np.int64) * num_nodes + df_edges['v'].values)

    # -------------------------------------------------------
    # B. Data Splitting (Edge Masking)
    # -------------------------------------------------------
    print("\n--- Splitting Data (Edge Masking) ---")
    mask_test = np.random.rand(len(df_edges)) < 0.2
    df_train_pos = df_edges[~mask_test].copy()
    df_test_pos = df_edges[mask_test].copy()
    
    print(f"Train Positives: {len(df_train_pos)}")
    print(f"Test Positives: {len(df_test_pos)}")
    
    # -------------------------------------------------------
    # C. Sparse Matrix Creation (CSR)
    # -------------------------------------------------------
    row = df_train_pos['u'].values
    col = df_train_pos['v'].values
    data = np.ones(len(df_train_pos)) 
    
    graph_csr_directed = sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # Symmetrized Graph for Topology
    graph_csr_sym = graph_csr_directed + graph_csr_directed.T
    graph_csr_sym.data = np.ones_like(graph_csr_sym.data) 
    
    degrees = np.array(graph_csr_sym.sum(axis=1)).flatten()
    
    # Weight Statistics
    weights_train = df_train_pos['weight'].values
    graph_csr_weighted = sparse.csr_matrix((weights_train, (row, col)), shape=(num_nodes, num_nodes))
    
    sum_out = np.array(graph_csr_weighted.sum(axis=1)).flatten()
    sum_in = np.array(graph_csr_weighted.T.sum(axis=1)).flatten()
    global_avg_weight = np.mean(weights_train)
    
    weight_stats = {
        'sum_out': sum_out,
        'sum_in': sum_in,
        'global_avg': global_avg_weight
    }

    # -------------------------------------------------------
    # D. Negative Sampling 
    # -------------------------------------------------------
    print("\n--- Negative Sampling (with filtering) ---")
    num_pos = len(df_train_pos)
    num_neg = num_pos * 5
    
    neg_u, neg_v = generate_negatives(num_neg, num_nodes, existing_edge_set)
    
    # -------------------------------------------------------
    # E. Feature Construction & Training
    # -------------------------------------------------------
    print("\n--- Building Training Features ---")
    X_pos = build_features(df_train_pos['u'].values, df_train_pos['v'].values, graph_csr_sym, node_props, degrees, weight_stats)
    y_pos = np.ones(len(X_pos))
    
    X_neg = build_features(neg_u, neg_v, graph_csr_sym, node_props, degrees, weight_stats)
    y_neg = np.zeros(len(X_neg))
    
    X_train = pd.concat([X_pos, X_neg], ignore_index=True)
    y_train = np.concatenate([y_pos, y_neg])
    
    y_train_weights_log = np.log1p(df_train_pos['weight'].values)
    
    print("\n--- Training Stage 1: Link Prediction (LightGBM) ---")
    # Note: LightGBM ignores columns in categorical_feature that aren't in input
    # but it's safer to check intersection if needed. 
    # Here we assume features constructed are present.
    model_existence = lgb.LGBMClassifier(objective='binary', n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1)
    model_existence.fit(X_train, y_train, categorical_feature=cat_features)
    
    print("\n--- Validating... ---")
    # Quick validation
    num_test_neg = len(df_test_pos)
    test_neg_u, test_neg_v = generate_negatives(num_test_neg, num_nodes, existing_edge_set)
    X_test_pos = build_features(df_test_pos['u'].values, df_test_pos['v'].values, graph_csr_sym, node_props, degrees, weight_stats)
    X_test_neg = build_features(test_neg_u, test_neg_v, graph_csr_sym, node_props, degrees, weight_stats)
    X_test = pd.concat([X_test_pos, X_test_neg], ignore_index=True)
    y_test = np.concatenate([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])
    
    probs = model_existence.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"Stage 1 ROC-AUC: {auc:.4f}")

    print("\n--- Training Stage 2: Weight Prediction (LightGBM Regressor) ---")
    model_weight = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1)
    model_weight.fit(X_pos, y_train_weights_log, categorical_feature=cat_features)
    
    # -------------------------------------------------------
    # H. Full Graph Prediction (All Unknown Edges)
    # -------------------------------------------------------
    predict_all_unknown_edges(
        num_nodes, existing_edge_set, graph_csr_sym, node_props, 
        degrees, weight_stats, model_existence, model_weight, 
        output_file=output_csv, threshold=threshold
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Link Prediction & Weight Regression Pipeline')
    parser.add_argument('--input', type=str, help='Path to input CSV file (columns: u, v, weight)', default=None)
    parser.add_argument('--features', type=str, help='Path to node features CSV file (columns: node_id, feat1, feat2...)', default=None)
    parser.add_argument('--output', type=str, help='Path to output CSV file for predictions', default='predicted_edges.csv')
    parser.add_argument('--threshold', type=float, help='Probability threshold for saving predicted edges', default=0.5)
    
    args = parser.parse_args()
    
    run_pipeline(input_csv=args.input, features_csv=args.features, output_csv=args.output, threshold=args.threshold)
