import subprocess
import sys
import datetime

"""script which finds the age in days of commits in a file
Input: path to repository, text file containing commits
Output: age of each commit in days
"""

MONTHS = {'Jan': 1,
          'Feb': 2,
          'Mar': 3,
          'Apr': 4,
          'May': 5,
          'Jun': 6,
          'Jul': 7,
          'Aug': 8,
          'Sep': 9,
          'Oct': 10,
          'Nov': 11,
          'Dec': 12
}


def processCommitDate(dateline):
    dateline = dateline.replace('   ', ' ')
    dateline = dateline.split(' ')
    date = datetime.datetime(int(dateline[5]), MONTHS[dateline[2]], int(dateline[3]))
    return date


def getAgesInDays(commits, directory, exclude=[]):
    cmd = "git log"
    log = subprocess.check_output(cmd.split(' '), cwd=directory).decode()
    newest = datetime.datetime(1,1,1)
    oldest = datetime.datetime.now()
    for commit in log.split('\n'):
        if commit.startswith('Date'):
            date = processCommitDate(commit)
            if date < oldest:
                oldest = date
            if date > newest:
                newest = date
    commits = open(commits, encoding='utf-8')
    ages = []
    for commit in commits.readlines():
        if commit.startswith('Date'):
            date = processCommitDate(commit)
            age = (date - oldest).days
            ages.append(str(age))
        if commit.startswith('commit '):
            exclude.append(commit.split(' ')[1].replace('\n', ''))
    return '\n'.join(ages)



if __name__ == "__main__":
    repo = sys.argv[1]
    commits = sys.argv[2]
    print(getAgesInDays(commits, repo))