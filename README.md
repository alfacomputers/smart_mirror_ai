# Smart Mirror AI

Minimal emotion-detection project for a smart mirror. Contains scripts to train and predict emotions and a small web/upload flow.

## Quick install

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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
