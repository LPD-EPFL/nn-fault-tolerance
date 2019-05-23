#!/bin/bash
file=$1
if [ "X$file" == "X" ]
then
	echo "Usage: $0 file.zip"
	exit 1
fi
mkdir temp_check
cd temp_check
rm -rf *
unzip -q ../$file -d .

# use lowecase here
bad_words="volodin epfl sergei mahdi andrey andrei sergey rachid guerraoui dcl distributed"
for a in *
do
  for word in $bad_words
  do
    if (( $(cat $a|grep -vaE "\"image/png\": \"[^\"]+\",\$"|tr '[:upper:]' '[:lower:]'|grep --color=ALWAYS -a "$word"|wc -l) > 0 ))
    then
      echo "FILE $a found word $word"
    fi
  done
done
cd ..
rm -rf temp_check
