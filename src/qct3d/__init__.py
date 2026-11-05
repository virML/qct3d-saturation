"""Code for "Quantum or just quantised? Angle-encoder saturation confounds ansatz and placement
comparisons in hybrid models for 3D CT".

Modules
-------
config         paths for every notebook, read from config.yaml
preprocessing  resample -> crop/pad -> HU windowing, shared by both cohorts
models         circuit families, adapters, classical controls, training, metrics
protocol       patient-grouped splits, arms, paired comparisons, result logs, noise model

`models` and `protocol` need PennyLane and PyTorch; `preprocessing` needs SimpleITK and PyTorch.
They are imported on demand, so `import qct3d` alone has no heavy dependencies.
"""
__version__ = "1.0.0"
