import torch
import argparse

def view_pt_file(file_path):
    """
    Loads a .pt file and prints a summary of its contents.
    """
    try:
        print(f"Loading model from: {file_path}")
        # Load the checkpoint onto the CPU
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
        
        print("\nFile type:", type(checkpoint))
        
        if isinstance(checkpoint, dict):
            # This is common for checkpoints that save more than just the model weights
            print("\nKeys in checkpoint:", checkpoint.keys())
            
            # Try to find the model's state dictionary
            model_state_dict = None
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            elif all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
                # It might be a raw state dictionary
                model_state_dict = checkpoint
            
            if model_state_dict:
                print("\n--- Model Layers and Parameters ---")
                for param_tensor, tensor_value in model_state_dict.items():
                    print(f"{param_tensor:<50} {list(tensor_value.size())}")
                print("------------------------------------")
            
            # Print other metadata if it exists
            print("\n--- Other Metadata ---")
            for key, value in checkpoint.items():
                if key not in ['model', 'state_dict']:
                    if isinstance(value, dict):
                        print(f"Key '{key}':")
                        for sub_key, sub_value in value.items():
                            print(f"  - {sub_key}: {sub_value}")
                    else:
                        print(f"Key '{key}': {value}")
            print("----------------------")

        else:
            # This might be the model object itself
            print("\n--- Model Architecture ---")
            print(checkpoint)
            print("--------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View the contents of a PyTorch .pt file.")
    parser.add_argument('file_path', type=str, help='The path to the .pt file.')
    
    args = parser.parse_args()
    
    view_pt_file(args.file_path)
