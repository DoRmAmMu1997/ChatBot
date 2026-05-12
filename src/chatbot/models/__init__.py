"""Model code for Aurora (~72B dense omni-modal) and Forge (~460B MoE coder).

Both models share the building blocks under ``models.common`` (RMSNorm,
RoPE+YaRN, GQA, MLA, SwiGLU, MoE, KV cache, decoder block), the vision
tower under ``models.vision``, and the audio subsystem (encoder + codec)
under ``models.audio``.

Note on directory names: the legacy package directories are still
``aurora_50b/`` and ``forge_250b/`` because Python imports are part of
the public API — renaming would force every caller to update. The YAML
files under ``configs/models/`` use the current sizes
(``aurora-72b.yaml`` / ``forge-460b.yaml``).
"""
