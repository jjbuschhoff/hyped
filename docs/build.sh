sphinx-apidoc -f -e -o source/api ../src/hyped --tocfile hyped
make clean
make html
