from subprocess import call


def convert_notebook_and_clean_imports(notebook_path):
    converted_script_path = notebook_path.replace('.ipynb', '.py')
    call(['jupyter', 'nbconvert', '--to', 'script', notebook_path])

    call(['autoflake', '--in-place', '--remove-unused-variables', converted_script_path])
