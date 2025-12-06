import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from sklearn.metrics import roc_auc_score, mean_squared_error, classification_report
import gc

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
    
    print(f"Starting Adamic-Adar calculation for {n_pairs} pairs...")
    
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
        node_props: Array where index = node_ID, value = category_code.
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
    
    # --- Node Property Features ---
    
    # 3. Categorical Match & Raw Categories
    u_cats = node_props[u_nodes]
    v_cats = node_props[v_nodes]
    feat_same_cat = (u_cats == v_cats).astype(int)
    
    feature_dict = {
        'pref_attach': feat_pref_attach,
        'adamic_adar': feat_adamic,
        'cat_u': u_cats,
        'cat_v': v_cats,
        'same_cat': feat_same_cat,
    }

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

def run_pipeline(num_nodes=60000, num_edges=1800000):
    print(f"--- Initializing Graph with {num_nodes} nodes and ~{num_edges} edges ---")
    
    # -------------------------------------------------------
    # A. Synthetic Data Generation (Replace with real data loading)
    # -------------------------------------------------------
    # Random edges
    sources = np.random.randint(0, num_nodes, num_edges)
    targets = np.random.randint(0, num_nodes, num_edges)
    # Remove self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    
    # Random weights (Power law-ish: most are 1, some are high)
    weights = np.random.zipf(2.0, size=len(sources))
    
    # Random Node Properties (Categorical, e.g., 3 categories)
    node_props = np.random.randint(0, 3, num_nodes)
    
    # Create dataframe
    df_edges = pd.DataFrame({'u': sources, 'v': targets, 'weight': weights})
    
    # Deduplicate edges (keep first or sum weights - simplified here to keep first)
    df_edges = df_edges.drop_duplicates(subset=['u', 'v'])
    print(f"Final edges after cleanup: {len(df_edges)}")
    
    # --- Create Set of Existing Edges for Fast Lookup ---
    # Using integer encoding: u * num_nodes + v for memory efficiency and speed
    print("Indexing existing edges for negative sampling...")
    existing_edge_set = set(df_edges['u'].values.astype(np.int64) * num_nodes + df_edges['v'].values)

    # -------------------------------------------------------
    # B. Data Splitting (Edge Masking)
    # -------------------------------------------------------
    print("\n--- Splitting Data (Edge Masking) ---")
    # Mask 20% of existing edges for validation/testing of "Existence"
    mask_test = np.random.rand(len(df_edges)) < 0.2
    
    df_train_pos = df_edges[~mask_test].copy()
    df_test_pos = df_edges[mask_test].copy()
    
    print(f"Train Positives: {len(df_train_pos)}")
    print(f"Test Positives: {len(df_test_pos)}")
    
    # -------------------------------------------------------
    # C. Sparse Matrix Creation (CSR)
    # -------------------------------------------------------
    # Create CSR from TRAINING edges only (to avoid leakage)
    row = df_train_pos['u'].values
    col = df_train_pos['v'].values
    data = np.ones(len(df_train_pos)) # Binary for topology
    
    # Directed Graph for specific directed features if needed
    graph_csr_directed = sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # Symmetrized Graph for Adamic-Adar (Topology)
    # Note: This sums weights if edges exist in both directions, but we just need non-zero structure
    graph_csr_sym = graph_csr_directed + graph_csr_directed.T
    graph_csr_sym.data = np.ones_like(graph_csr_sym.data) # Binarize
    
    # Pre-calculate degrees (from symmetric graph usually best for undirected metrics)
    degrees = np.array(graph_csr_sym.sum(axis=1)).flatten()
    
    # Pre-calculate Weight Statistics for Bayesian Smoothing
    # Using directed graph weights
    # Create weighted CSR
    weights_train = df_train_pos['weight'].values
    graph_csr_weighted = sparse.csr_matrix((weights_train, (row, col)), shape=(num_nodes, num_nodes))
    
    sum_out = np.array(graph_csr_weighted.sum(axis=1)).flatten()
    # For sum_in, we transpose
    sum_in = np.array(graph_csr_weighted.T.sum(axis=1)).flatten()
    global_avg_weight = np.mean(weights_train)
    
    weight_stats = {
        'sum_out': sum_out,
        'sum_in': sum_in,
        'global_avg': global_avg_weight
    }

    # -------------------------------------------------------
    # D. Negative Sampling (Smart Ratio 1:5)
    # -------------------------------------------------------
    print("\n--- Negative Sampling (with filtering) ---")
    num_pos = len(df_train_pos)
    num_neg = num_pos * 5
    
    neg_u, neg_v = generate_negatives(num_neg, num_nodes, existing_edge_set)
    
    # -------------------------------------------------------
    # E. Feature Construction (Train Set)
    # -------------------------------------------------------
    print("\n--- Building Training Features ---")
    
    # Positive Samples
    X_pos = build_features(
        df_train_pos['u'].values, 
        df_train_pos['v'].values, 
        graph_csr_sym, 
        node_props,
        degrees,
        weight_stats
    )
    y_pos = np.ones(len(X_pos))
    
    # Negative Samples
    X_neg = build_features(
        neg_u, 
        neg_v, 
        graph_csr_sym, 
        node_props,
        degrees,
        weight_stats
    )
    y_neg = np.zeros(len(X_neg))
    
    # Combine
    X_train = pd.concat([X_pos, X_neg], ignore_index=True)
    y_train = np.concatenate([y_pos, y_neg])
    
    # Weights for regression (only for positives)
    y_train_weights = df_train_pos['weight'].values
    # Log transform targets
    y_train_weights_log = np.log1p(y_train_weights)
    
    # Categorical features indices for LightGBM
    cat_features = ['cat_u', 'cat_v', 'same_cat']
    
    # -------------------------------------------------------
    # F. Stage 1: Existence Model (Classification)
    # -------------------------------------------------------
    print("\n--- Training Stage 1: Link Prediction (LightGBM) ---")
    
    model_existence = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1
    )
    
    model_existence.fit(
        X_train, y_train,
        categorical_feature=cat_features
    )
    
    # Validate on Test Set (Positives + Random Negatives)
    print("\n--- Validating... ---")
    # Generate test negatives
    num_test_neg = len(df_test_pos)
    test_neg_u, test_neg_v = generate_negatives(num_test_neg, num_nodes, existing_edge_set)
    
    X_test_pos = build_features(df_test_pos['u'].values, df_test_pos['v'].values, graph_csr_sym, node_props, degrees, weight_stats)
    X_test_neg = build_features(test_neg_u, test_neg_v, graph_csr_sym, node_props, degrees, weight_stats)
    
    X_test = pd.concat([X_test_pos, X_test_neg], ignore_index=True)
    y_test = np.concatenate([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])
    
    probs = model_existence.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"Stage 1 ROC-AUC: {auc:.4f}")

    # -------------------------------------------------------
    # G. Stage 2: Property Model (Weight Regression)
    # -------------------------------------------------------
    print("\n--- Training Stage 2: Weight Prediction (LightGBM Regressor) ---")
    
    # Train ONLY on positive examples
    # X_pos corresponds to df_train_pos
    
    model_weight = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1
    )
    
    model_weight.fit(
        X_pos, y_train_weights_log,
        categorical_feature=cat_features
    )
    
    # Evaluate on Test Positives
    y_test_weights_log = np.log1p(df_test_pos['weight'].values)
    preds_log = model_weight.predict(X_test_pos)
    
    # Calculate RMSLE (on log scale)
    rmsle = np.sqrt(mean_squared_error(y_test_weights_log, preds_log))
    print(f"Stage 2 RMSLE (Log Scale Error): {rmsle:.4f}")
    
    # -------------------------------------------------------
    # H. Integrated Inference Example
    # -------------------------------------------------------
    print("\n--- Inference Example on New Pairs ---")
    # Simulate 5 candidates
    cand_u = np.random.randint(0, num_nodes, 500*500)
    cand_v = np.random.randint(0, num_nodes, 500*500)
    
    X_cand = build_features(cand_u, cand_v, graph_csr_sym, node_props, degrees, weight_stats)
    
    # 1. Predict Existence
    exist_probs = model_existence.predict_proba(X_cand)[:, 1]
    
    # 2. Predict Weight (for all, but we'd usually filter)
    weight_preds_log = model_weight.predict(X_cand)
    weight_preds_real = np.expm1(weight_preds_log)
    
    results = pd.DataFrame({
        'u': cand_u,
        'v': cand_v,
        'prob_exists': exist_probs,
        'pred_weight': weight_preds_real
    })
    
    # Filter by threshold
    threshold = 0.5
    likely_edges = results[results['prob_exists'] > threshold]
    
    print("All Candidates Predictions:")
    print(results)
    print(f"\nLikely Edges (Threshold > {threshold}):")
    print(likely_edges)

if __name__ == "__main__":
    # Run the full pipeline
    run_pipeline()
