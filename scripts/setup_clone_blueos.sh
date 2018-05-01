#!/bin/bash

module load gcc/4.9.3
module load cmake/3.9.2
module load spectrum-mpi/2017.11.10
module load cuda/9.0.176

function is_git_dir() {
  if [ -d .git ]; then
    echo .git;
  else
    git rev-parse --git-dir 2> /dev/null;
  fi;
}

function create_branch() {

git submodule foreach -q --recursive \
    'branch="$(git config -f $toplevel/.gitmodules submodule.$name.branch)"; \
     [ "$branch" = "" ] && \
     git checkout master || git checkout $branch && \
     git submodule update --init' 
}

if "is_git_dir"; then
  echo "error .. inside git repo, please change directory"
else
  echo "creating directory Laghos-submodules-fresh"
  mkdir -p "Laghos-submodules-fresh"
  cd "Laghos-submodules-fresh"
  pwd
  git clone git@github.com:jonesholger/Laghos.git --branch feature/blueos --recursive
  cd "Laghos"
  create_branch
fi;  

exit 0


