import shutil

with open('../problist.txt') as f:
    lines = f.read().splitlines()
lines = [x[:-3] for x in lines]

for file in lines:
    shutil.move("~/collections/" + file, "../data/miplib/" + file)