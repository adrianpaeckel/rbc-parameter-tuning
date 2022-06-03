cd rbc-parameter-tuning/experiment
echo Start
while 'true'
do 
python rbc_soc_ref.py
echo CRASH: Restarting experiment
sleep 5
done

