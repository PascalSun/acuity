"""Utility commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import click

from talk2metadata.cli.utils import get_yaml_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.group(name="utils")
def utils_group():
    """Utility commands."""


@utils_group.group(name="agent")
def agent_group():
    """Manage LLM agent providers and servers."""


@agent_group.command(name="vllm-server")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
def vllm_server_cmd(config: str | None):
    """Start a vLLM OpenAI-compatible API server.

    All configuration is read from the run config YAML file.

    \b
    Examples:
        # Start server using config settings
        talk2metadata utils agent vllm-server

        # Start with a specific config file
        talk2metadata utils agent vllm-server --config configs/wamex.yml
    """
    try:
        import vllm  # noqa: F401
    except ImportError:
        click.echo(
            "❌ vLLM is not installed.\n" "   Install it with: pip install vllm",
            err=True,
        )
        sys.exit(1)

    cfg = get_yaml_config(config)
    agent_config = cfg.get("agent", {})
    vllm_config = agent_config.get("vllm", {})

    model = vllm_config.get("model") or agent_config.get("model")
    if model is None:
        click.echo(
            "❌ Model not found in config.\n"
            "   Please set agent.vllm.model in the config YAML",
            err=True,
        )
        sys.exit(1)

    host = vllm_config.get("host", "0.0.0.0")

    port = vllm_config.get("port")
    if port is None:
        base_url = vllm_config.get("base_url", "")
        if base_url:
            try:
                parsed = urlparse(base_url)
                if parsed.port:
                    port = parsed.port
                elif parsed.scheme == "http":
                    port = 80
                elif parsed.scheme == "https":
                    port = 443
                else:
                    port = 8000
            except Exception:
                port = 8000
        else:
            port = 8000

    tensor_parallel_size = vllm_config.get("tensor_parallel_size")
    gpu_memory_utilization = vllm_config.get("gpu_memory_utilization")
    max_model_len = vllm_config.get("max_model_len")
    dtype = vllm_config.get("dtype")
    trust_remote_code = vllm_config.get("trust_remote_code", False)
    download_dir = vllm_config.get("download_dir")
    api_key = vllm_config.get("api_key")
    served_model_name = vllm_config.get("served_model_name")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]

    if tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if dtype:
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if download_dir:
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--download-dir", str(download_path)])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    click.echo("🚀 Starting vLLM server...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Endpoint: http://{host}:{port}/v1")
    click.echo(f"\n   Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Server stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        click.echo(f"\n❌ Server failed to start: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        sys.exit(1)
