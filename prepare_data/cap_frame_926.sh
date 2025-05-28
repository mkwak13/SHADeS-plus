# Navigate to the base directory
cd ../Datasets/BBPS-2-3Frames/Frames/ #Inpainted_gen9

# Use a for loop to iterate over all directories
for dir in */; do
  echo "Processing directory: $dir"
  
  # Navigate into each directory
  cd "$dir"
  
  # Correctly find, echo, and remove files with numbers greater than 926
  find . -type f -name '*.jpg' | while read file; do
    num=$(echo "$file" | sed 's/^.*\///; s/^0*//; s/[^0-9]*$//')
    num=$((10#$num))
    if [ "$num" -gt 926 ]; then
      echo "Removing file: $file"
      rm -- "$file"
    fi
  done
  
  # Go back to the base directory
  cd ..
done