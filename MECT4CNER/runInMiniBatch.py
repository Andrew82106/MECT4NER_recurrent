import os

maxx = 15197

for i in range(0, maxx, 1400):
    os.system(f"python MECT4CNER/predict.py --stage {i}")