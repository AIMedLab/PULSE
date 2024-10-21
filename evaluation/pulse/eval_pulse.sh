bash bench_eval_main.sh -m pulse -d code15-test
bash bench_eval_main.sh -m pulse -d mmmu-ecg
bash bench_eval_main.sh -m pulse -d ptb-test
bash bench_eval_main.sh -m pulse -d ptb-test-report
bash bench_eval_main.sh -m pulse -d ptb-valid
bash bench_eval_main.sh -m pulse -d cpsc-test-fix
bash bench_eval_main.sh -m pulse -d g12-test-no-cot
bash bench_eval_main.sh -m pulse -d csn-test-no-cot
bash bench_eval_main.sh -m pulse -d ecgqa-test

# evaluate arena
bash bench_eval_arena.sh -m pulse -d arena