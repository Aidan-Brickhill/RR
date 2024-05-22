# Setting Up Your Atari RL Environment - Steps by Sayf

## Step 1: Create a Conda Environment

First, create a new Conda environment named `atari` with Python 3.10.

```bash
conda create --name atari python=3.10
```

Activate the newly created environment:

```bash
conda activate atari
```

## Step 2: Install pipx

Next, install `pipx` using Python's `pip` module and ensure the path is set correctly.

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

## Step 3: Install Poetry

Use `pipx` to install Poetry, a dependency management tool.

```bash
pipx install poetry
```

## Step 4: Clone the CleanRL Repository

Clone the CleanRL repository from GitHub:

```bash
git clone git@github.com:SuperSayf/cleanrl.git
```

Navigate into the cloned repository directory:

```bash
cd cleanrl
```

## Step 5: Install CleanRL Dependencies

Install the project dependencies using Poetry:

```bash
poetry install
```

Upgrade and install PyTorch with CUDA 11.3 support:

```bash
poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
```

Install additional dependencies for Atari and JAX support:

```bash
poetry install -E "atari jax"
```

Upgrade JAX to the specified version with CUDA 11 and cuDNN 8.2 support:

```bash
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Step 6: Run CleanRL Example

Finally, run the CleanRL example script for the Atari environment `BreakoutNoFrameskip-v4`:

```bash
poetry run python cleanrl/dqn_atari_jax.py --env-id BreakoutNoFrameskip-v4
```

---

<!-- go onto a node and use as if its a computer
srun -N 1 -p stampede --pty bash -->

conda create --name handover

conda activate handover

git clone https://github.com/Aidan-Brickhill/RR.git

pip install stable-baselines3[extra]

python3 train.py

source ~/.handover