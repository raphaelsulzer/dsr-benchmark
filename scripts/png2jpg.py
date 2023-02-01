import os
from PIL import Image


path = "/home/rsulzer/overleaf/SurveyAndBenchmark/figures"
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            if not "input" in file:
                # print(file)
                # im1 = Image.open(os.path.join(root,file))
                # im1 = im1.convert('RGB')
                # im1.save(os.path.join(root,file).split('.')[0]+".jpg")
                os.remove(os.path.join(root,file))

