#!/bin/bash

n=3
let pmax=n-1
script=$1

rm results*
rm *.png
rm *.out*

for i in `seq 0 $pmax`
do
  cmd="PATH=/localhome/volodin/miniconda3/bin:$PATH\nsource activate neuronfailure\npython $script $n $i 2>&1|tee $script.out$i\ntelegram-send \"Process $i exited with last line \`tail -n 1 $script.out$i\`\""
  echo -e "#!/bin/bash\n$cmd" > run$i.sh
  screen -d -m bash run$i.sh
done
