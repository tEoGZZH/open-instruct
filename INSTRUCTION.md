# Enviroment Setup
## Install with uv
In open-instuct, [uv](https://docs.astral.sh/uv/) is used to setting up the environment. Use `curl` to install uv:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have `curl`, you can use `wget`:

```
wget -qO- https://astral.sh/uv/install.sh | sh
```

Set up the environmnet use:

```
uv sync .
```

## Install with Docker
You can also build a Docker image using the Dockerfile. This approach is mentioned in the original README, although it hasn’t been tested personally.

```
docker build . \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
  --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
  -t open_instruct_dev
```

# Model Training

Traning scripts for data mixing with llama3.1-8B model are stored at:
```
/home/master_thesis/open-instruct/scripts/train/itcvs/data_mxing/llama3.1-8B
```

The structure is for the RWTH CLAIX cluster; if you’re using a different cluster, you may need to modify it accordingly.

There are scripts covering data proportions from 0% to 100% (in 10% increments). Training on 0%, 30%, 50%, 70%, and 100% should be sufficient. Please run a smoke test before executing other training scripts.

The training dataset is available on Google Drive:
[Dataset link](https://drive.google.com/file/d/1DqAvBzA-iGgicxhAMty8fnm1lWene9cP/view?usp=drive_link)
