import pandas as pd
import argparse
import os

def filter_k_core(df, k):
    """Iteratively filter dataframe to obtain k-core."""
    iteration = 0
    while True:
        num_users_before = df['user_id'].nunique()
        num_items_before = df['item_id'].nunique()
        
        # User core filter
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
        
        # Item core filter
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]
        
        num_users_after = df['user_id'].nunique()
        num_items_after = df['item_id'].nunique()
        
        if num_users_before == num_users_after and num_items_before == num_items_after:
            break
        iteration += 1
        print(f"Iteration {iteration}: Users {num_users_after}, Items {num_items_after}, Rows {len(df)}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Apply iterative K-Core filtering to Goodreads CSV datasets.")
    parser.add_argument("--profiles_in", default="data/profiles.csv", help="Path to unfiltered profiles.csv")
    parser.add_argument("--interactions_in", default="data/interactions.csv", help="Path to unfiltered interactions.csv")
    parser.add_argument("--k", type=int, default=20, help="k value for k-core filtering (default: 20)")
    parser.add_argument("--out_dir", default="data/k_core", help="Output directory for filtered files")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading unfiltered datasets...")
    if not os.path.exists(args.profiles_in) or not os.path.exists(args.interactions_in):
        print(f"Error: Unfiltered files '{args.profiles_in}' or '{args.interactions_in}' not found.")
        print("Please run goodreads_preprocess.py first to extract the full datasets.")
        return
        
    df_prof = pd.read_csv(args.profiles_in)
    df_inter = pd.read_csv(args.interactions_in)
    
    print(f"Original shapes:")
    print(f"- Profiles: {df_prof.shape}")
    print(f"- Interactions: {df_inter.shape}")
    
    # Filter interactions using k-core
    print(f"\nFiltering interactions with k={args.k}...")
    filtered_inter = filter_k_core(df_inter, args.k)
    
    # Align profiles with filtered interactions
    print("\nAligning profiles with filtered interactions...")
    valid_pairs = set(zip(filtered_inter['user_id'].astype(str), filtered_inter['item_id'].astype(str)))
    
    df_prof['user_id_str'] = df_prof['user_id'].astype(str)
    df_prof['item_id_str'] = df_prof['item_id'].astype(str)
    df_prof['pair'] = list(zip(df_prof['user_id_str'], df_prof['item_id_str']))
    
    filtered_prof = df_prof[df_prof['pair'].isin(valid_pairs)].drop(columns=['pair', 'user_id_str', 'item_id_str'])
    
    profiles_out = os.path.join(args.out_dir, f"profiles_k{args.k}.csv")
    interactions_out = os.path.join(args.out_dir, f"interactions_k{args.k}.csv")
    
    filtered_prof.to_csv(profiles_out, index=False)
    filtered_inter.to_csv(interactions_out, index=False)
    
    print(f"\n[Success] Filtered dataset saved to '{args.out_dir}':")
    print(f"- Profiles: {filtered_prof.shape} -> {profiles_out}")
    print(f"- Interactions: {filtered_inter.shape} -> {interactions_out}")
    print(f"Unique Users: {filtered_inter['user_id'].nunique()}, Unique Items: {filtered_inter['item_id'].nunique()}")

if __name__ == "__main__":
    main()
