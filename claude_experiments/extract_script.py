"""Extract notebook code cells to a runnable script for headless verification.
Strips %/! magics (packages already installed in the venv) and forces Agg backend."""
import nbformat, sys
nb = nbformat.read("multilingual_consensus_colab.ipynb", as_version=4)
out = ['import matplotlib; matplotlib.use("Agg")\n',
       'import os; os.environ["MPLBACKEND"]="Agg"\n']
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    out.append(f"\n# ===== cell {i} =====\n")
    for line in c.source.splitlines():
        s = line.strip()
        if s.startswith("%") or s.startswith("!"):
            out.append("# [magic stripped] " + line + "\n")
        else:
            out.append(line + "\n")
out.append('\nimport torch as _t\n')
out.append('print("PEAK_GPU_MEM_MiB", round(_t.cuda.max_memory_allocated()/1048576))\n')
with open("nb_script.py", "w") as f:
    f.writelines(out)
print("wrote nb_script.py")
