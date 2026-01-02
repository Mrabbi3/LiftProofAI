"""
LiftProof.Ai Configuration
"""

# Model path
MODEL_PATH = "models/my_trained_model.pt"

# Camera settings (for when you connect real camera)
CAMERAS = [
    {
        "name": "Store_Camera",
        "brand": "hikvision",
        "ip": "192.168.1.64",
        "port": 554,
        "username": "admin",
        "password": "password",
        "channel": 1,
        "enabled": False  # Set to True when you have camera
    }
]

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
POCKET_DISTANCE_THRESHOLD = 0.15
POCKET_TIME_THRESHOLD = 45
PROCESS_EVERY_N_FRAMES = 2

# Save evidence
SAVE_EVIDENCE_CLIPS = True
EVIDENCE_PATH = "data/evidence/"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/liftproof.log"