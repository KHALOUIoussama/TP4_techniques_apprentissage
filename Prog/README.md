## How to
```
cd prog
sudo pip install virtualenv      # This may already be installed
virtualenv -p python3 .env       # Create a virtual environment (python3)
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies

# Work on the assignment for a while ...

deactivate                       # Exit the virtual environment
```