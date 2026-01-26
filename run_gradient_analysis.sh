# Experiment: Gradient Analysis on Sokoban (1-step)

# Single GPU run
python3 train.py --config-name "_2_sokoban" \
    trainer.project_name='AGEN_gradient_analysis' \
    trainer.experiment_name='gradient_analysis_sokoban' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.total_training_steps=1 \
    algorithm.adv_estimator=gae \
    system.CUDA_VISIBLE_DEVICES=\'0,1,2,3\' \
    trainer.val_before_train=False \
    +trainer.gradient_analysis_mode=True \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +critic.model.override_config.attn_implementation=eager \
    > gradient_analysis.log 2>&1
