#!/bin/bash

conda create --name=fake_news python=3.10.12
conda activate fake_news
pip install --no-cache-dir -r requirements.txt
