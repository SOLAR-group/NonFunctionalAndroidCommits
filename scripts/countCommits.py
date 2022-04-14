import os
import sys
"""
A Script to count the number of commit messages in each file in a directoey

Input: directory to analyse (consisting of text files containing commits)
Output: The number of commits in each text file and the number of unique commit hashes across the directory
"""

wd = sys.argv[1]

hshs = {}
for f in os.scandir(wd):
    if 'txt' in f.path:
        hshs[f.path] = []
        lines = open(f,'r')
        for line in lines.readlines():
            if line.startswith('commit'):
                hshs[f.path].append(
                    line.split(' ')[1].replace('\n','')
                    )
uniqHsh = set()
for f in hshs:
    uniqHsh = uniqHsh.union(hshs[f])
    print(f,len(hshs[f]))    

print(len(uniqHsh))