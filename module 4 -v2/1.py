import torch
import os

def count_parameters_in_file(file_path):
    """Loads a PyTorch checkpoint and returns the total number of parameters."""
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # Extract the state_dict from common checkpoint structures
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Recursively count parameters if the state dict is nested
        def recurse_count(obj):
            total = 0
            if isinstance(obj, dict):
                for val in obj.values():
                    total += recurse_count(val)
            elif isinstance(obj, (list, tuple)):
                for val in obj:
                    total += recurse_count(val)
            elif torch.is_tensor(obj):
                total += obj.numel()
            return total
        
        total_params = recurse_count(state_dict)
        
        # Convert to millions for easier reading
        params_in_millions = total_params / 1_000_000
        return total_params, params_in_millions
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def main():
    # --- IMPORTANT: List the paths to your .pt files here ---
    # Add the full paths to your three model files
    model_files = {
        "My_Model_1": "../module 3/cmsegnet_stage2.pt",
        "My_Model_2": "cmsegnet_stage2_head.pt",
        "My_Model_3": "cmsegnet_stage2.pt"
    }
    
    print("\n" + "="*50)
    print("MODEL PARAMETER COMPARISON")
    print("="*50)
    print(f"{'Model Name':<20} {'Params':<15} {'Params (M)':<15}")
    print("-"*50)
    
    for name, file_path in model_files.items():
        if not os.path.exists(file_path):
            print(f"{name:<20} {'File not found':<15}")
            continue
        
        total_params, params_in_M = count_parameters_in_file(file_path)
        if total_params is not None:
            # Human-readable formatting
            readable_params = f"{total_params:,}"
            readable_M = f"{params_in_M:.2f}"
            print(f"{name:<20} {readable_params:<15} {readable_M:<15}M")
        else:
            print(f"{name:<20} {'Error':<15}")

if __name__ == "__main__":
    main()