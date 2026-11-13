"""Training entry points the manager launches under the LeRobot venv's python, BY FILE PATH.

They live in their own folder on purpose: a script's folder is first on sys.path, and rd_manager itself has a
datasets.py that would shadow Hugging Face's `datasets` for everything lerobot imports.
"""
