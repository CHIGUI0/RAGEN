#!/bin/bash
bash scripts/runs/smoke_test_for_ke.sh --groups 1,4,5 --kl-type kl
bash scripts/runs/smoke_test_for_ke.sh --groups 1,4,5 --kl-type mse
bash scripts/runs/smoke_test_for_ke.sh --groups 1,2,3,4,5 --kl-type low_var_kl
