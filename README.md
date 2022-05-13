# AttentionSleepiness for CHASE2022
Project goal:
Applying attention mechanism in Voiceome dataset to determine sleepiness sates of speaker with HuBERT embedding.

## Bash Command
#### No-Attention mechaninsm
```bash
python3 train.py --task=1
```
Use *--task=0* to train the model with all tasks' data

#### With Attention mechanism
```bash
python3 train.py --attention=1
```



## Speech tasks
| Task   | Response columns | Description | ICASSP2022's Result |
| ------ | ---------------- | ------------ | ------------------ |
| Task 1 | response1        | Microphone test | 69.70% |
| Task 2 | response2        | Free speech | 77.24% |
| Task 3 | response3        | Picture description | 70.66% |
| Task 4 | response4        | Category nameing | 75.0% |
| Task 5 | response5        | Phonemic fluency | 78.34% |
| Task 6 | response6        | Paragraph reading | 73.14% |
| Task 7 | response7        | Sustained phonation | 77.68% |
| Task 8 | response8        | Diadochokinetic (puh-puh-puh)| 67.61% |
| Task 9 | response9        | Diadochokinetic (puh-tuh-kuh) | 69.83% |
| Task 10| response10,..., response34 | Confrontational naming | 81.13% |
| Task 11| response35,..., response44| Non-word pronuciation | 78.66% |
| Task 12| response46, response48 | Memory recall | 80.87% |
| **All task** | _all above_ | | **81.29%** |
