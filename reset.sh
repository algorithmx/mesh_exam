#!/bin/bash

rm -rf __pycache__/
rm -rf */__pycache__/
rm -rf */*/__pycache__/
rm -rf */*/*/__pycache__/


rm -rf .pytest_cache/
rm -rf */.pytest_cache/
rm -rf */*/.pytest_cache/
rm -rf */*/*/.pytest_cache/