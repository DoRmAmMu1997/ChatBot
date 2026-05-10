from src.chatbot.config import ModelConfig, chatbot_10b_config, tiny_config
from src.chatbot.params import estimate_parameter_count


def test_chatbot_10b_config_is_about_ten_billion_parameters():
    config = chatbot_10b_config()
    report = estimate_parameter_count(config)

    assert config.model_name == "chatbot-10b"
    assert config.n_layer == 36
    assert config.n_embd == 5120
    assert config.n_head == 40
    assert config.n_kv_head == 8
    assert 9.99e9 <= report.total <= 10.01e9


def test_yaml_config_round_trip_for_tiny_config():
    config = ModelConfig.from_yaml_file("configs/chatbot-tiny.yaml")

    assert config.to_dict() == tiny_config(vocab_size=128).to_dict()
