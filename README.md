# Smart Mirror AI

Minimal emotion-detection project for a smart mirror. Contains scripts to train and predict emotions and a small web/upload flow.

## Quick install

On Linux, the recommended bootstrap script is:

```bash
./scripts/bootstrap.sh
```

To install the Ubuntu/Debian system packages as well, run:

```bash
sudo ./scripts/bootstrap.sh --install-system
```

Then activate the environment:

```bash
source .venv/bin/activate
```

> Note: On older CPUs without AVX support (for example Intel Celeron N4000), official TensorFlow wheels may fail with an illegal instruction. The project can still bootstrap, but model loading may require a compatible CPU or a custom TensorFlow build.

For traditional pip (slower):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Git setup and push

Initialize git, add files, and push to a remote (replace `<REMOTE_URL>`):

```powershell
git init
git add .
git commit -m "Initial project import"
git remote add origin <REMOTE_URL>
git branch -M main
git push -u origin main
```

Note: Large model files (`*.h5`) and the `uploads/` or `dataset/` folders are ignored by default. Remove the patterns in `.gitignore` if you want to track them.
