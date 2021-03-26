#/bin/bash

rm -r ./docs/*
pdoc3 --html PyPruning -c latex_math=True --force --template-dir pdoc-templates --output-dir docs
mv ./docs/PyPruning/* ./docs/
rm -r ./docs/PyPruning
