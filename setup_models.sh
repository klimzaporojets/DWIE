#!/usr/bin/env bash

python -m spacy download en_core_web_sm

wget -O data.zip "https://cloud.ilabt.imec.be/index.php/s/mS34ZgN2m5gBEzD/download?files=data.zip"
unzip data.zip
rm data.zip
