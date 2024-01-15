
import re

def remove_comments_docstrings_empty_lines_from_python_file(file_content):
    # Remove comments and docstrings from Python file
    file_content_without_comments = re.sub(r'(?:(?<!\\)(?<!\'|\")#.*)|^\s*#.*', '', file_content, flags=re.MULTILINE)
    file_content_without_comments_and_docstrings = re.sub(r'(\'\'\'(.*?)(\'\'\'|\"\"\")|\"\"\"(.*?)(\'\'\'|\"\"\")|\'.*?\'|\".*?\")', '', file_content_without_comments, flags=re.DOTALL)
    # Remove empty lines
    file_content_without_empty_lines = re.sub(r'^\s*\n', '', file_content_without_comments_and_docstrings, flags=re.MULTILINE)
    return file_content_without_empty_lines

def remove_backticks(input_string):
    return input_string.replace("`", "")
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


# Read template file
template_file_path = 'template.js'
with open(template_file_path, 'r') as template_file:
    template_content = template_file.read()


python_file_contents = []

for item in files_to_add:
    file = item['file']
    placeholder = item['placeholder']

    # Read Python file
    with open(file, 'r') as file:
        python_file_content = file.read()
        

    python_file_content = remove_backticks(python_file_content)
    python_file_content = re.sub(r'(?m)^.*from\s+\..*?import\s+.*$', '', python_file_content)
    python_file_content = re.sub(r'\bclass\s+Points2Regions\b', 'class Points2RegionClass', python_file_content)

    # Remove comments from Python file
    #python_file_content_without_comments = remove_comments_docstrings_empty_lines_from_python_file(python_file_content)
    python_file_contents.append(python_file_content)

    # Find the indentation of the placeholder
    match = re.search(r'(\s*)' + placeholder, template_content)
    if match:
        indentation = match.group(1)
    else:
        indentation = ''

    # Replace placeholder in template with Python content maintaining the same indentation
    template_content = template_content.replace(placeholder, f'{indentation}{python_file_content}')

# Write the result to Points2Region_build.js
output_file_path = r'C:\Users\axela\.tissuumaps\plugins\Points2Regions.js'
with open(output_file_path, 'w') as output_file:
    output_file.write(template_content)


# Write Python contents to Points2Region_build.py
python_output_file_path = r'C:\Users\axela\.tissuumaps\plugins\Points2Regions.py'
with open(python_output_file_path, 'w') as python_output_file:
    python_output_file.write('\n\n'.join(python_file_contents))