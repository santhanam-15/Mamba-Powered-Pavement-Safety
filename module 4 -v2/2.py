import torch

checkpoint = torch.load("cmsegnet_stage2.pt", map_location="cpu")

# Check what type it is
print(type(checkpoint))

# If it's a state_dict (dictionary of tensors)
if isinstance(checkpoint, dict):
    # Check keys to see structure
    print("Keys:", list(checkpoint.keys())[:5])  # first few keys
    
    # Count parameters from state dict
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    else:
        sd = checkpoint
    
    total_params = sum(p.numel() for p in sd.values())
    print(f"Total parameters: {total_params:,}")
    
# If it's a full model object (less common)
else:
    # Try to get state_dict from model object
    try:
        sd = checkpoint.state_dict()
        total_params = sum(p.numel() for p in sd.values())
        print(f"Total parameters: {total_params:,}")
    except:
        print("Unable to determine parameters. Loaded object type:", type(checkpoint))