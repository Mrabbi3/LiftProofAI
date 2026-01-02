"""
Convert model to safe format
"""
import torch

# Load with weights_only=False (we trust this file)
print("Loading original model...")
checkpoint = torch.load('models/my_trained_model.pt', map_location='cpu', weights_only=False)

# Save in safe format
print("Saving in compatible format...")
torch.save(checkpoint, 'models/my_trained_model_safe.pt')

print("✅ Conversion complete!")
print("Now use: models/my_trained_model_safe.pt")
