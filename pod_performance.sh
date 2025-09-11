# performance
for TASK in cola
do
for LORA in "" "--lora"
do
echo "Running: python3 spikelora_finetuning/performance.py --task $TASK --project performance $LORA"
python3 spikelora_finetuning/performance.py --task $TASK --project performance $LORA > logs/performance_${TASK}_${LORA//--/--}.log 2>&1
done
done
