import os


def setup() -> None:
    os.environ["SLURM_JOB_NAME"] = "bash"
    set_port()


def set_port() -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    try:
        default_port = os.environ["SLURM_JOB_ID"]
        default_port = default_port[-4:]

        # All ports should be in the 10k+ range
        default_port = int(default_port) + 15000

    except Exception:
        default_port = 12910

    os.environ["MASTER_PORT"] = str(default_port)
