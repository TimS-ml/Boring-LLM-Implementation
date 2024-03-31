find "./" -iname '*.py' -type f -exec sed -i -e \
    "s/broing/boring/g;\
     " {} \;

