import os


def get_env_variable(default_key: str, default_value: str) -> str:
    if not os.environ.get(default_key):
        os.environ[default_key] = default_value
        return default_value
    else:
        return os.environ[default_key]

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY", "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")