#!/bin/bash
echo "Sometimes there are stderr outputs printed, and they contain the home directory name (volodin) which should not be there for an anonymous submission"
lst=$(ls|grep -vE "\.sh$")
grep --color=ALWAYS volodin $lst

for a in $lst
do
	if [ "$(cat $a|grep volodin|wc -l)" != "0" ]
	then
		echo "Pruning $a"
		cat $a | grep -v "volodin" > ${a}.tmp
		mv ${a}.tmp $a
	fi
done
