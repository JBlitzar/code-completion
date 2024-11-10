#!/bin/bash


pattern="events.out.tfevents.*.local.*.*"


find . -type f -name "$pattern" | while read -r file; do
    if [[ "$file" == *"masked_hostname"* ]]; then
        #echo "Skipping $file (already masked)"
        continue
    fi

    dir=$(dirname "$file")
    filename=$(basename "$file")
    
    new_filename=$(echo "$filename" | sed -E 's/(events\.out\.tfevents\.[0-9]+.)[0-9a-zA-Z\-]+(\.local\.[0-9]+\.[0-9]+)/\1masked_hostname\2/')
    

    mv "$file" "$dir/$new_filename"
    echo "Renamed $file to $dir/$new_filename"
done

echo "Killing python..."
killall python

echo "Killing caffeinate..."
killall caffeinate