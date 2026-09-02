# Codec Schema Evolution

This unreleased project resets to one canonical `bar_tensor_schema.v2` runtime. The compatibility authority is an `encoding_manifest.json` whose schema name, hashes, shapes, grid policy, parser/control provenance, and source identity policy must agree with the archive and index. Old artifacts fail fast; Git history, tags, and published reports preserve experiments. Legacy runtime code is not retained as an importable archive.
