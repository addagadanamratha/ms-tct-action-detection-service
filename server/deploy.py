r"""
Deploy server v5 to the Hugging Face Space (run from D:\mstct\space_v5 after login).
    python deploy_v5.py
"""
from huggingface_hub import HfApi

SPACE = "namratha-addagada/ms-tct-action-detection"
api = HfApi()
api.add_space_variable(SPACE, "MODEL_REPO", "namratha-addagada/ms-tct-charades-fusion")
api.add_space_variable(SPACE, "MAX_VIDEO_SECONDS", "30")
api.upload_folder(folder_path=".", repo_id=SPACE, repo_type="space",
                  allow_patterns=["app.py", "Dockerfile", "requirements.txt", "web/*"],
                  commit_message="v7: confidence (likely vs uncertain), label + hint fixes")
print("Uploaded. Watch the build: https://huggingface.co/spaces/" + SPACE)
print("App: https://namratha-addagada-ms-tct-action-detection.hf.space")
