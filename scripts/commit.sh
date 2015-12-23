#!/bin/bash

# DBN-IRIS Commit script
rm -Rrf *~

git add .

read -e -p "Commit Comment: " comment
git commit -m "$comment"
git push origin master

