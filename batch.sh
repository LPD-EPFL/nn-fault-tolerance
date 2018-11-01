#!/bin/bash

n=3
let pmax=n-1
script=$1

for i in `seq 0 $pmax`
do
  cmd="PATH=/localhome/volodin/miniconda3/bin:$PATH\nsource activate neuronfailure\npython $script $n $i 2>&1|tee $script.out$i"
  echo -e "#!/bin/bash\n$cmd" > run$i.sh
  screen -d -m bash run$i.sh
done
