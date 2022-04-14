import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import ast
import time
import sys

path = sys.argv[1]
def getHtml(url):
        try:
            fp = urllib.request.urlopen(link)
            return fp
        except urllib.error.HTTPError as e:
            if e.getcode() == 429:
                time.sleep(5)
                return getHtml(url)
    

for line in open(path, "r").readlines():
    hsh = line.strip().split(" ")[1]
    link = "https://github.com/search?q={}&type=commits".format(hsh)
    fp = getHtml(link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()

    soup = BeautifulSoup(mystr, features="html.parser")

    for hyper in soup.find_all("a", {"class": "message markdown-title js-navigation-open"}):
        for attrib in hyper["data-hydro-click"].split(","):
            tokens = attrib.split(":")
            if tokens[0] == "\"url\"":
                print(":".join(tokens[1:]).replace("\"","").replace("}",""))
        break
    time.sleep(2)
