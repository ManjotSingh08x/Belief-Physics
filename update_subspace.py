import re

with open("experiments/04-subspace-analysis.py", "r") as f:
    code = f.read()

# Add MAX_T
code = code.replace("K = params.DEFAULT_N  # K=10", "K = params.DEFAULT_N  # K=10\nMAX_T = 20\nNUM_BLOCKS = (SEQ_LEN - MAX_T) // K + 1")

# Update cache_activations
old_cache = """def cache_activations(model, data_iter, num_batches, desc="Caching"):
    cache_X = {l: {t: [] for t in range(K)} for l in range(NUM_LAYERS)}
    cache_y_belief = []
    cache_y_vel = []

    for _ in tqdm(range(num_batches), desc=desc):
        batch = next(data_iter)
        theta_bins = torch.clamp(torch.round(batch["theta"] * (180.0 / np.pi)), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        vel_chunks = chunk_batch(batch["velocity"].to(device), SEQ_LEN + K)
        belief_chunks = chunk_batch(batch["belief_state"].to(device), SEQ_LEN + K)
        
        for c_idx in range(len(coord_chunks)):
            inputs = coord_chunks[c_idx][:, :-K]
            
            # The target at the end of each K-block is index 0 of the *next* block conceptually,
            # or index K-1 of the current block. Based on original script, it was index 0.
            # We want to predict the Belief State and Velocity constant for that block.
            belief_targets = belief_chunks[c_idx][:, :-K, :].reshape(-1, SEQ_LEN//K, K, 3)[:, :, 0, :].reshape(-1, 3)
            vel_targets = vel_chunks[c_idx][:, :-K].reshape(-1, SEQ_LEN//K, K)[:, :, 0].reshape(-1, 1)
            
            cache_y_belief.append(belief_targets.cpu())
            cache_y_vel.append(vel_targets.cpu())
            
            with torch.no_grad():
                model(inputs)
            
            for l in range(NUM_LAYERS):
                # Shape: (N, 250, 128)
                act_l = model.activations[f'layer{l}']
                # Reshape to (N, 25, 10, 128)
                act_l = act_l.reshape(act_l.shape[0], SEQ_LEN // K, K, EMBED_DIM)
                
                for t in range(K):
                    # For a specific relative offset t, grab all N * 25 instances
                    act_lt = act_l[:, :, t, :].reshape(-1, EMBED_DIM)
                    cache_X[l][t].append(act_lt.cpu())

    # Concatenate caches
    final_cache_X = {l: {t: torch.cat(cache_X[l][t], dim=0) for t in range(K)} for l in range(NUM_LAYERS)}
    final_y_belief = torch.cat(cache_y_belief, dim=0)
    final_y_vel = torch.cat(cache_y_vel, dim=0)
    
    return final_cache_X, final_y_belief, final_y_vel"""

new_cache = """def cache_activations(model, data_iter, num_batches, desc="Caching"):
    cache_X = {l: {t: [] for t in range(MAX_T)} for l in range(NUM_LAYERS)}
    cache_y_belief = []
    cache_y_vel = []

    for _ in tqdm(range(num_batches), desc=desc):
        batch = next(data_iter)
        theta_bins = torch.clamp(torch.round(batch["theta"] * (180.0 / np.pi)), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        vel_chunks = chunk_batch(batch["velocity"].to(device), SEQ_LEN + K)
        belief_chunks = chunk_batch(batch["belief_state"].to(device), SEQ_LEN + K)
        
        for c_idx in range(len(coord_chunks)):
            inputs = coord_chunks[c_idx][:, :-K]
            
            # Extract targets specifically for the NUM_BLOCKS
            # We want to match the activations from block b (starting at b*K)
            b_targets = []
            v_targets = []
            for b in range(NUM_BLOCKS):
                b_targets.append(belief_chunks[c_idx][:, b * K, :])
                v_targets.append(vel_chunks[c_idx][:, b * K])
            
            # Shapes: (N, NUM_BLOCKS, 3) and (N, NUM_BLOCKS)
            b_targets = torch.stack(b_targets, dim=1).reshape(-1, 3)
            v_targets = torch.stack(v_targets, dim=1).reshape(-1, 1)
            
            cache_y_belief.append(b_targets.cpu())
            cache_y_vel.append(v_targets.cpu())
            
            with torch.no_grad():
                model(inputs)
            
            for l in range(NUM_LAYERS):
                # Shape: (N, 250, 128)
                act_l = model.activations[f'layer{l}']
                
                for t in range(MAX_T):
                    # We extract the activation at t steps after the start of each block
                    # For all b in 0..NUM_BLOCKS-1: indices are b*K + t
                    indices = [b * K + t for b in range(NUM_BLOCKS)]
                    act_lt = act_l[:, indices, :].reshape(-1, EMBED_DIM)
                    cache_X[l][t].append(act_lt.cpu())

    # Concatenate caches
    final_cache_X = {l: {t: torch.cat(cache_X[l][t], dim=0) for t in range(MAX_T)} for l in range(NUM_LAYERS)}
    final_y_belief = torch.cat(cache_y_belief, dim=0)
    final_y_vel = torch.cat(cache_y_vel, dim=0)
    
    return final_cache_X, final_y_belief, final_y_vel"""

code = code.replace(old_cache, new_cache)

# Replace all subsequent K with MAX_T in the loops, except where it says 'for t in range(K):'
code = code.replace("for t in range(K):", "for t in range(MAX_T):")
code = code.replace("np.zeros((NUM_LAYERS, K))", "np.zeros((NUM_LAYERS, MAX_T))")
code = code.replace("NUM_LAYERS * K", "NUM_LAYERS * MAX_T")
code = code.replace("np.zeros(K)", "np.zeros(MAX_T)")
code = code.replace("range(K)", "range(MAX_T)")

# Make Layer Sweep use a single T
l_sweep_old = """    print("\\n--- PHASE 4: L Sweep (Timesteps Concatenated) ---")
    l_belief_r2 = np.zeros(NUM_LAYERS)
    l_vel_r2 = np.zeros(NUM_LAYERS)
    l_angle = np.zeros(NUM_LAYERS)
    for l in range(NUM_LAYERS):
        X_tr_l = torch.cat([X_train[l][t] for t in range(MAX_T)], dim=1)
        X_te_l = torch.cat([X_test[l][t] for t in range(MAX_T)], dim=1)
        
        b_beta, b_mux, b_muy = solve_ridge(X_tr_l, y_train_belief, alpha)
        l_belief_r2[l] = evaluate_r2(b_beta, b_mux, b_muy, X_te_l, y_test_belief)
        
        v_beta, v_mux, v_muy = solve_ridge(X_tr_l, y_train_vel, alpha)
        l_vel_r2[l] = evaluate_r2(v_beta, v_mux, v_muy, X_te_l, y_test_vel)
        
        l_angle[l] = subspace_angle(b_beta, v_beta)"""

l_sweep_new = """    print("\\n--- PHASE 4: L Sweep (Single T) ---")
    PEAK_T = 9  # Use t=9 for layer sweep
    l_belief_r2 = np.zeros(NUM_LAYERS)
    l_vel_r2 = np.zeros(NUM_LAYERS)
    l_angle = np.zeros(NUM_LAYERS)
    for l in range(NUM_LAYERS):
        X_tr_l = X_train[l][PEAK_T]
        X_te_l = X_test[l][PEAK_T]
        
        b_beta, b_mux, b_muy = solve_ridge(X_tr_l, y_train_belief, alpha)
        l_belief_r2[l] = evaluate_r2(b_beta, b_mux, b_muy, X_te_l, y_test_belief)
        
        v_beta, v_mux, v_muy = solve_ridge(X_tr_l, y_train_vel, alpha)
        l_vel_r2[l] = evaluate_r2(v_beta, v_mux, v_muy, X_te_l, y_test_vel)
        
        l_angle[l] = subspace_angle(b_beta, v_beta)"""

code = code.replace(l_sweep_old, l_sweep_new)
code = code.replace("L-Sweep, 1280D", "L-Sweep, t=9, 128D")

with open("experiments/04-subspace-analysis.py", "w") as f:
    f.write(code)

print("Replaced!")
