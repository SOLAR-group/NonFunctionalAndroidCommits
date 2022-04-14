import subprocess
import sys
"""
Script to get the diff for a patrricular commit
Inputs: repo (path to repository) Commit hash (hash of commit to aqquire diff for)
output: diff of commit
"""
def getDiff(commitHash,repo):
    cmd = 'git diff ' + commitHash + '^ ' + commitHash

    log = subprocess.check_output(cmd.split(' '), cwd=repo).decode()
    return log


repo = sys.argv[1]
hsh = sys.argv[2]

print(getDiff(hsh, repo))