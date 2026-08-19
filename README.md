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
pip install scikit-learn
pip install numpy pandas pillow matplotlib tqdm openpyxl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python scikit-image seaborn


# run
python src\preprocessing\analyze_dataset.py
python src\preprocessing\create_split.py
