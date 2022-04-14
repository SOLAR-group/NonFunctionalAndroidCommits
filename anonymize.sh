for f in $(find . -print | grep -i .txt);
    do
    echo "$(grep "^commit" "$f")" > "$f";
    done