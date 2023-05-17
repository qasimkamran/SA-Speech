import gdown


README_FILE = 'README.md'


def get_notebook_drive_links():
    notebook_links = []
    flag = '### Notebooks'
    flag_found = False

    with open(README_FILE, 'r') as file:
        for line in file:
            line = line.strip()

            print(line)

            if flag_found and line.startswith('- https://colab.research.google.com/drive/'):
                notebook_links.append(line)

            if line == flag:
                flag_found = True
            elif line and not line.startswith('#'):
                flag_found = False

    return notebook_links


if __name__ == '__main__':
    notebook_links = get_notebook_drive_links()
    for link in notebook_links:
       gdown.download(link)
