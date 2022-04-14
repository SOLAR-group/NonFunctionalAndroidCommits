for f in $(find . -print | grep -i .txt);
    do
    echo "$(grep -v "^Author\|^Date" "$f")" > $f;
    done