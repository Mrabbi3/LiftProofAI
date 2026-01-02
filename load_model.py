import torch
from ultralytics import YOLO

# Allow loading of your custom trained model
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.PoseModel'])

model = YOLO('models/my_trained_model.pt')
print("✅ Model loaded successfully!")
