cd rbc-parameter-tuning/experiment
echo Start
while 'true'
do 
python rbc_3_1_2.py
echo CRASH: Restarting experiment
sleep 5
done

