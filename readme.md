# FastAPI Backend

1. Run this in Ubuntu WSL 
2. Install Python, Pip, Python3-venv
3. Create Virtual Environment 
```
python3 -m venv <name>
```
4. Activate your Virtual Environment
```
source <env_name>/bin/activate
```
5. Run "pip install -r requirements.txt"
6. Install Detectron2
7. Download the model for Layout Parser https://drive.google.com/drive/folders/1q0rYE0g7h1Fy4PCB368KXJXa6Der9UlI?usp=sharing
8. Create .env for private keys
9. When running the app "uvicorn main:app --host 0.0.0.0 --port 8001 --loop asyncio"


# Installing Detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
