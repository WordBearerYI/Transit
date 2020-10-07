for i in {1..30}
do
echo ${i}
sed -i "s/traj[0-9]*/traj${i}/g" ./run_eval_vis_AVD.sh
sh ./lstm_run_train_AVD.sh $i > 15_h${i}_avlog.txt
wait
done
