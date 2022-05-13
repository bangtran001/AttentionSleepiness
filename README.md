# AttentionSleepiness for CHASE2022
Project goal:
Applying attention mechanism in Voiceome dataset to determine sleepiness sates of speaker with HuBERT embedding.

## Bash Command
```bash
python3 train.py --task=1
```
Use *--task=0* to train the model with all tasks' data

## Speech tasks
| Task   | Response columns | Description |
| ------ | ---------------- | ------------ |
| Task 1 | response1        | Microphone test |
| Task 2 | response2        | Free speech |
| Task 3 | response3        | Picture description |
| Task 4 | response4        | Category nameing |
| Task 5 | response5        | Phonemic fluency |
| Task 6 | response6        | Paragraph reading |
| Task 7 | response7        | Sustained phonation |
| Task 8 | response8        | Diadochokinetic (puh-puh-puh)|
| Task 9 | response9        | Diadochokinetic (puh-tuh-kuh) |
| Task 10| response10,..., response34 | Confrontational naming |
| Task 11| response35,..., response44| Non-word pronuciation |
| Task 12| response46, response48 | Memory recall |
