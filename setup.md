
## Table of Contents

- [Clone the repository](#clone-the-repository)
- [Setup environment](#setup-environment)
- [Run pre-commit hooks manually](#run-pre-commit-hooks-manually)
- [Submodule commands](#submodule-commands)
- [Compilation](#compilation)
- [Check](#check)
- [Koyeb ssh](#koyeb-ssh)

# Clone the repository

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
```

# Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

# Run pre-commit hooks manually

```bash
pre-commit run --all-files
```

# Submodule commands

```bash
# update submodule to the latest commit on the remote branch
git submodule update --remote

# check status
git submodule status
```

The tt-metal submodule is pinned to the `stable` branch in `.gitmodules`.

>[!WARNING]
>Destruction alert! The following commands will change the structure and all team members will need to use `git submodule update --remote` to update their local submodule to the new branch. Make sure to inform your team about the change.

```bash
# Change submodule branch
cd third-party/tt-metal
git checkout <other-branch>
cd ../..
git add third-party/tt-metal
git commit -m "Update tt-metal to different branch"
```

```bash
[skip ci] maybe used in commit message to skip CI if chore changes only
```

# Compilation

```bash
-
```

## Check

```bash
-
```

# Koyeb ssh

<https://github.com/koyeb/tenstorrent-examples/tree/main/tt-ssh>

```bash
ssh -p <PORT> root@<IP_ADDRESS>
```
