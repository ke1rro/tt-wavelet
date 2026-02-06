
# Setup enviroment

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

>[!WARNING]
>Destruction alert! The following commands will change the structure and all team members will need to use `git submodule update --remote` to update their local submodule to the new branch. Make sure to inform your team about the change.

```bash
# Change submodule branch
cd fwt_lib/third-party/tt-metal
git checkout <other-branch>
cd ..
git add fwt_lib/third-party/tt-metal
git commit -m "Update tt-metal to different branch"
```

```bash
[skip ci] maybe used in commit message to skip CI if chore changes only
```

# Compilation

```bash
git submodule update --init --recursive

mkdir -p build && cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DTT_METAL_BUILD_TESTS=OFF -DBUILD_PROGRAMMING_EXAMPLES=OFF -DWITH_PYTHON_BINDINGS=OFF -DENABLE_CCACHE=ON

ninja tt_metal
ninja tt_wavelet_test
```

## Check

```bash
export TT_METAL_RUNTIME_ROOT=/work/third-party/tt-metal
export TT_METAL_HOME=/work/third-party/tt-metal
./tt-wavelet/tt_wavelet_test
```
