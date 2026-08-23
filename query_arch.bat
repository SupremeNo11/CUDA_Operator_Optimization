@echo off
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>nul
pause
