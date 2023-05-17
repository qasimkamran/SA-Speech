import gdown


README_FILE = 'README.md'


# Checks the README.md for the notebook flag and gets a list of all retrievable links
def get_notebook_drive_links():
    notebook_links = []
    flag = 'notebooks'
    flag_found = False

    with open(README_FILE, 'r') as file:
        for line in file:
            line = line.strip()

            if flag_found and line.startswith('http'):
                notebook_links.append(line)

            if line == flag:
                flag_found = True
            elif line and not line.startswith('#'):
                flag_found = False

    return notebook_links


if __name_- == '__main__':
    for link in notebook_links:
       gdown.download(link)
