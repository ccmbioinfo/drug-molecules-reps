#!/bin/bash

#Loop through all files that match the pattern features_*_*.pt
for file in features_labels_labels_*_*.pt; do
    # Remove the second "labels_" from the filename
    new_file=$(echo "$file" | sed 's/labels_labels_/labels_/')
    
    # Rename the file
    mv "$file" "$new_file"
    
    # Print the renaming action
    echo "Renamed $file to $new_file"
done
