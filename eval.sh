cuda=0
object_name="nonstop"
container_name="plate"
pick_place_ckpt_path="/ML-vePFS/tangyinzhou/yinuo/dp_train_zhiting/ckpts/20250919_114059/policy_step_30000_seed_0.ckpt"
pick_place_obs_normalize_params_path="/ML-vePFS/tangyinzhou/yinuo/dp_train_zhiting/ckpts/20250917_165837/norm_stats_1_epsnum_500.pkl"
CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="diffusion_policy" --ckpt_path="${pick_place_ckpt_path}" \
    -e "TabletopPickPlaceEnv-v1" -s 0 --num-episodes 300 --num-envs 25 --save-video --max_episode_len 400 \
    --object_name="$object_name" --container_name="$container_name" -r panda_wristcam -c pd_ee_pose \
    --obs_normalize_params_path="$pick_place_obs_normalize_params_path" --initial_eps_count 5050