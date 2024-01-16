
import re

files_to_add = [
    {
        'file' : '../points2regions/utils.py',
        'placeholder' : '{UTILS_HERE}'
    },
    {
        'file' : '../points2regions/geojson.py',
        'placeholder' : '{GEOJSON_HERE}'
    },
    {
        'file' : '../points2regions/points2regions.py',
        'placeholder' : '{P2R_HERE}'
    }

]



python_file_contents = []
for item in files_to_add:
    # Read Python file
    with open(item['file'], 'r') as file:
        python_file_content = file.read()
    
    # Remove relative imports
    python_file_content = re.sub(r'(?m)^.*from\s+\..*?import\s+.*$', '', python_file_content)
    
    # Change name of main class (not sure if we actually need this)
    python_file_content = re.sub(r'\bclass\s+Points2Regions\b', 'class Points2RegionClass', python_file_content)
    python_file_contents.append(python_file_content)

# Write Python contents to Points2Regions.py
python_output_file_path = r'points2region_tmap.py'
with open(python_output_file_path, 'w') as python_output_file:
    python_output_file.write('\n\n'.join(python_file_contents))