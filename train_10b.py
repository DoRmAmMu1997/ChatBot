"""Training entrypoint for the original untrained ChatBot-10B config.

This script intentionally does not ship weights. It wires the normal training
loop to ``configs/chatbot-10b.yaml`` so a user with suitable multi-GPU hardware
can initialize and train the model outside GitHub.
"""

from src.chatbot.train_10b import main


if __name__ == "__main__":
    main()
