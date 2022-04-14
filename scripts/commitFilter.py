import argparse
import subprocess
import re
"""
Python3 script which takes a git repo and a regular expression.

Returns a file containing all commits with lines tht match the regular expression.
"""

if __name__ == "__main__":
    # Parse arguments 
    parser = argparse.ArgumentParser(description='Search git log for regex.')
    parser.add_argument('directory', metavar='D', type=str,
                   help='path to the repository')
    parser.add_argument('regex', metavar='R', type=str,
                   help='regex containing keywords to extract')
    parser.add_argument('--out', metavar=('O'), type=str,
                   help='Output file for matching commits')
    args = parser.parse_args()
    # get all commits from repository
    cmd = "git log"
    log = subprocess.check_output(cmd.split(' '), cwd=args.directory).decode()
    commit = ""
    regex = re.compile(args.regex)
    commits = ""
    contreg = False
    # Check each line for regex match
    for line in log.split('\n'):
        # After parsing a whole commit, add it to output if regex matches
        if line.startswith("commit"):
            if contreg:
                commits += commit + '\n'
                contreg = False
            commit = ""
        #Check if regex matches
        if regex.search(line, flags=re.I) != None:
            contreg = True
        commit += line + '\n'
    if args.out != None:
        out = open(args.out, 'w', encoding="utf-8")
    else:
        out = open('out.txt', 'w', encoding="utf-8")
    out.write(commits)
    out.close()