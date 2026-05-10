"""Main entrypoint for chatting with the upgraded small LLM.

Run this after training a checkpoint:

    python chatbot.py --checkpoint checkpoints/chatbot-small-llm.pt
"""

from src.chatbot.chat import main


if __name__ == "__main__":
    main()
