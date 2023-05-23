import gdown
import shutil
import io
import re


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


def download_notebook(share_link):
    file_id = share_link.split("/drive/")[1].split("/")[0]
    file_id = file_id.split("?")[0]
    download_link = 'https://drive.google.com/uc?id='+ file_id
    filename = gdown.download(download_link)
    destination = 'notebooks/' + filename
    shutil.move(filename, destination)

if __name__ == '__main__':
    notebook_links = get_notebook_drive_links()
    for link in notebook_links:
        download_notebook(link)
