"""
This program extracts .txt files of all the .html files from the directory,
using the BeautifulSoup module, into the folder "ExtractedText"
"""

# pylint: disable=C0103

import os
import time
import urllib.request
from bs4 import BeautifulSoup


def main():
    '''main'''
    directory = r"Z:\Assignment1\wiki-small\en\articles"
    start = time.time()
    traverse(directory)
    end = time.time()
    print("running time : " + str(end - start))


def traverse(direc):
    '''traverses through all the html files in the directory
    and calls the extract function for each file'''
    for root, dirs, files in os.walk(direc):
        for fname in files:
            #print("fname:", fname)
            current_file = "%s%s%s%s" % ("file:\\\\\\", os.path.abspath(root), os.path.sep, fname)
            current_file = current_file.replace("\\", "/")
            #print("src:", current_file)
            extract(current_file)


def extract(file):
    '''extracts the text from the given html doc,
    and writes into a new text file with the same name,
    in the ExtractedText folder'''
    url = file
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator=' ')

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    index = file.rfind("/")
    fname = "ExtractedText/" + file[index+1:-5] + ".txt"
    print(fname)
    f = open(fname, "w", encoding="utf-8")
    f.write(text)
    f.close()
    return


if __name__ == '__main__':
    main()
