# Apple_Leaf_Disease
Apple leaf disease classification and severity grading using deep learning

# virtual environment
py -3.11 -m venv .venv
# activate env
.\.venv\Scripts\Activate.ps1

(if there is an execution policy error run - 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
)

# installations
python.exe -m pip install --upgrade pip
pip install scikit-learn
pip install numpy pandas pillow matplotlib tqdm openpyxl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python scikit-image seaborn
pip install timm


# run
python src\preprocessing\analyze_dataset.py
python src\preprocessing\create_split.py
python src\classification\model.py

# requirements 
python -m pip install -r requirements.txt

# git user
git config --global user.name
git config --global user.email

git config user.name "New Name"
git config user.email "newemail@example.com"