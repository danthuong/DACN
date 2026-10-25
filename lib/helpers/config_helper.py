"""Configuration loading with deep inheritance and private .env expansion."""

import os
import re
from pathlib import Path

import yaml


_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")


def load_dotenv(path, override=False):
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in ("'", '"'):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    result = dict(base)
    for key, value in override.items():
        result[key] = _deep_merge(result[key], value) if key in result else value
    return result


def _expand_environment(value, missing):
    if isinstance(value, dict):
        return {key: _expand_environment(item, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, missing) for item in value]
    if not isinstance(value, str):
        return value
    match = _ENV_PATTERN.match(value)
    if match:
        name, default = match.groups()
        resolved = os.environ.get(name, default)
        if resolved is None:
            missing.add(name)
            return value
        # Preserve numeric/bool/null types for values that occupy a full field.
        return yaml.safe_load(resolved) if resolved != "" else ""

    def replace(match):
        name, default = match.groups()
        resolved = os.environ.get(name, default)
        if resolved is None:
            missing.add(name)
            return match.group(0)
        return resolved

    embedded_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
    return embedded_pattern.sub(replace, value)


def _load_with_extends(path):
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        # First resolve relative to the selected config, then repository cwd.
        local = path.parent / parent_path
        parent_path = local if local.is_file() else Path.cwd() / parent_path
    return _deep_merge(_load_with_extends(parent_path), payload)


def load_config(path, env_file=".env"):
    load_dotenv(env_file)
    config = _load_with_extends(path)
    missing = set()
    config = _expand_environment(config, missing)
    if missing:
        raise ValueError(
            "Missing required environment variables: {}. Add them to {}.".format(
                ", ".join(sorted(missing)), env_file))
    return config

