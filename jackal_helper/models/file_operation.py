import shutil
import os
"""
for f in os.listdir("."):
    if f.startswith("model"):
        sf = os.path.join(f, "map")
        tf = f.replace("model", "habitat")
        shutil.move(sf, tf)

for f in os.listdir("."):
    if f.startswith("habitat"):
        stl_file = os.path.join(f, "meshes", "map.stl")
        target_stl_file = os.path.join(f, "meshes", f + ".stl")
        if os.path.exists(stl_file):
            shutil.move(stl_file, target_stl_file)
"""
"""
for i in range(72):
    shutil.copy("sample/model.sdf", 'habitat%d/model.sdf' %i)

    replacements = {'map':'habitat%d' %i}

    lines = []
    with open(os.path.join('habitat%d' %i, "model.sdf")) as infile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            lines.append(line)
    with open(os.path.join('habitat%d' %i, "model.sdf"), 'w') as outfile:
        for line in lines:
            outfile.write(line)

    lines = []
    with open(os.path.join('habitat%d' %i, "model.config")) as infile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            lines.append(line)
    with open(os.path.join('habitat%d' %i, "model.config"), 'w') as outfile:
        for line in lines:
            outfile.write(line)
"""

for i in range(72):
    shutil.copy("../worlds/map.sdf", '../worlds/habitat%d.sdf' %i)
    replacements = {'map':'habitat%d' %i}
    lines = []
    with open('../worlds/habitat%d.sdf' %i) as infile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            lines.append(line)
    with open('../worlds/habitat%d.sdf' %i, 'w') as outfile:
        for line in lines:
            outfile.write(line)