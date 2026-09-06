# The circuit server (src/serve), for the k3s cluster: `uv sync` of the lock
# with the serve extra, no models inside. The HF cache and the results root
# are mounted at run time (see ~/m/projects/k8s/mi-lab). Torch comes from the
# lock with its CUDA wheels, so the plain Python base is enough; the GPU is
# reached through the nvidia runtime class.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
# A C compiler, because torch reaches one at *run* time on this GPU: some of
# its ops are Triton kernels, and Triton JITs them on first use by shelling out
# to `cc`. Without it the model loads, /health is green, and the first
# generation is the only thing that fails — 500 with "Failed to find C
# compiler" from inside a torch op, several frames below anything this repo
# wrote. `-slim` ships no compiler, so it has to be asked for.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# Only what the build itself reads. Runtime settings are set at the bottom, so
# changing one does not invalidate the `uv sync` layer below — that layer is
# most of the ~17 GiB image and re-resolving it costs a download of torch.
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra serve
COPY configs configs
COPY data data
COPY src src
COPY scripts scripts

# The weights arrive as a read-only mount and HF_HOME must not point into it:
# huggingface_hub reads its token at $HF_HOME/token and takes lock files beside
# the cache, so a read-only HF_HOME fails the load with a PermissionError on
# `.../huggingface/token` whose message blames a cancelled download and a stale
# lock — neither of which is what happened. So the *hub* cache is named on its
# own (HF_HUB_CACHE: the mount, only ever read) and HF_HOME stays on the
# container's writable filesystem. The Deployment sets these again, and must:
# these are the defaults for `docker run`, not the deployed configuration.
ENV HF_HOME=/tmp/hf HF_HUB_CACHE=/hf-hub HF_HUB_OFFLINE=1 \
    MI_LAB_CONFIG=qwen3-1.7b MI_LAB_CIRCUITS=/circuits MI_LAB_TASK=translation
EXPOSE 8000
CMD ["uv", "run", "--frozen", "--no-sync", "python", "-m", "scripts.serve"]
