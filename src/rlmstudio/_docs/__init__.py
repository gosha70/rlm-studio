"""Bundled copy of the repository's ``docs/`` for the Learn tab.

This package exists as a placeholder so that ``pip install rlm-studio`` always
creates the ``rlmstudio/_docs/`` directory. The build pipeline (see
``RELEASING.md``) copies ``docs/`` here before ``uv build`` runs, because an
installed wheel has no repository ``docs/`` to read.

If you are reading this file as the only contents of ``rlmstudio/_docs/``,
the docs have not been copied in, and the server falls back to the
repository's ``docs/`` (a source checkout).
"""
