import importlib.util, subprocess, sys

# Only pip-install missing packages — full resolve can take minutes every run.
_REQUIRED = [
    ('open_clip', 'open_clip_torch'),
    ('transformers', 'transformers'),
    ('datasets', 'datasets'),
    ('matplotlib', 'matplotlib'),
    ('PIL', 'Pillow'),
    ('scipy', 'scipy'),
]
_missing = [pip for mod, pip in _REQUIRED if importlib.util.find_spec(mod) is None]
if _missing:
    print('Installing missing:', _missing)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_missing], check=False)
else:
    print('Deps already installed — skipping pip.')
