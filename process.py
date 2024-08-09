from model import SRSelector
import json
import os


selector = SRSelector()

root = "/home/25e_zim@lab.graphicon.ru/mnt/calypso/25e_zim/SR+codec/subjectify"
stats = {}
for codec in os.listdir(root):
    for video in os.listdir(root + "/" + codec):
        for file in os.listdir(root + "/" + codec + "/" + video):
            result = selector(root + "/" + codec + "/" + video + "/" + file)
            stats[f"{codec}@{video}@{file}"] = result
           

with open("./results.json", 'w') as f:
    json.dump(stats, f, indent=4)
