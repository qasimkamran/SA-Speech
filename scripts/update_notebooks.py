import argparse
import gdown
import shutil
import io
import re
import os

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('--filename', dest='filename', action='store', required=False, type=str)

args = parser.parse_args()

README_FILE = 'README.md'


def get_notebook_drive_links():
    notebook_links = []
    with open(README_FILE, "r") as readme_file:
        readme_content = readme_file.read()
    notebooks_flag = re.search(r"### Notebooks\n", readme_content)

    if notebooks_flag:
        links = re.findall(r"- (.*?)\n", readme_content[notebooks_flag.end():])
        notebook_links = [link.strip() for link in links]

    return notebook_links


def download_notebook(share_link, filename=None):
    file_id = share_link.split("/drive/")[1].split("/")[0]
    file_id = file_id.split("?")[0]
    download_link = 'https://drive.google.com/uc?id='+ file_id
    temp_filename = gdown.download(download_link)

    if filename == temp_filename or filename is None:
        destination = 'notebooks/' + temp_filename
        shutil.move(filename, destination)
        print('Updated - {0}'.format(temp_filename))
    else:
        os.remove(temp_filename)
        print('Not Updated - {0}'.format(temp_filename))

if __name__ == '__main__':
    notebook_links = get_notebook_drive_links()
    filename = args.filename
    for link in notebook_links:
        download_notebook(link, filename)
