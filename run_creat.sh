#!/usr/bin/bash

var_corr=`find /home/yaser/yaser_temp/src/ff_corr -mindepth 1 -maxdepth 1`
var_dc=`find /home/yaser/yaser_temp/src/ff_dc -mindepth 1 -maxdepth 1`


cntr=0
for i in $var_corr
do
  echo "$(echo "#!/usr/bin/bash
cd /home/yaser/yaser_temp/src
python main.py -i $i -o /home/yaser/yaser_temp/output_refined_gu4_s5/ff_corr" > run_script_corr_$cntr.sh)"
((cntr+=1))
done

cntr=0
for i in $var_dc
do
  echo "$(echo "#!/usr/bin/bash
cd /home/yaser/yaser_temp/src
python main.py -i $i -o /home/yaser/yaser_temp/output_refined_gu4_s5/ff_dc" > run_script_dc_$cntr.sh)"
((cntr+=1))
done
