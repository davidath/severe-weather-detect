#! /bin/bash
# $1: Directory
# $2: String find
# $3: String replace
for i in $(find $1 -name '*.ini')
do
    sed -i "s/$2/$3/g" "$i"
done


