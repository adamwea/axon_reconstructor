if there are shared utilites / helpers / plotting type functions that would be applicable to two stages. Where do you put them in this schema? To avoid redundant code.
Thought for a couple of seconds

Put them at the highest level that is still semantically honest. In this schema, that usually means a pipeline-level shared package, not inside either stage. So I would add something like:

pipeline/
├── shared/
│   ├── __init__.py
│   ├── io.py
│   ├── paths.py
│   ├── logging.py
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── style.py
│   │   ├── chip.py
│   │   └── colors.py
│   ├── validation.py
│   └── utils.py
└── stages/
    ├── reconstruct/
    └── waveform_extraction/

The rule I would use is:

if it is only used by one stage, keep it in that stage
if it is used by two or more stages and is genuinely generic, move it to pipeline/shared/
if it is used by several stages but is specific to one domain, make a more specific shared package like:
pipeline/shared/plotting/
pipeline/shared/models/
pipeline/shared/hdmea/
pipeline/shared/templates/

I would not dump everything into one giant utils.py, because that becomes a junk drawer fast. Better to have small shared modules by concern.

A good practical split for your kind of code would be:

pipeline/shared/paths.py
path construction, output directory helpers, run naming
pipeline/shared/logging.py
logger setup, debug flag wiring, common log formatting
pipeline/shared/plotting/
shared figure setup, MEA/chip plotting helpers, common annotations, export helpers
pipeline/shared/io.py
generic serialization helpers, JSON/YAML reading/writing, maybe dataframe save/load wrappers
pipeline/shared/validation.py
common config checks, path checks, schema-like validation helpers
pipeline/shared/types.py or models/
only if multiple stages truly share structured objects

The main caution is this:

do not move something to shared just because it is duplicated once
first confirm the abstraction is actually stable
duplication is sometimes better than premature sharing

A good heuristic is: the second time you need the same logic, pause and ask whether the shared version would have a clear name independent of both stages. If yes, move it. If not, leave it local.

So in your schema, the normal place is:

pipeline/shared/

or, if narrower,

pipeline/shared/plotting/
pipeline/shared/io/
pipeline/shared/validation/

not inside one stage and imported sideways by another. Sideways imports between stages are usually what you want to avoid.