"""手动集成验证：使用本地 SmolVLA 权重检查缓存、动作采样及 PPO 梯度。

在项目根目录运行（需要 lerobot-env、CUDA 和本地权重）：
    python tests/test_flow_noise_smolvla_smoke.py
不会启动仿真或写入训练检查点；不同批大小的 BF16 数值偏差仅报告。
"""

def main():
    import numpy as np
    import torch
    from so101_lab.policies.rl.flow_noise_smolvla import FlowNoiseSmolVLA
    p=FlowNoiseSmolVLA('outputs/smolvla_iql_easy_v1',device='cuda')
    obs={'observation.state':np.zeros(6,dtype=np.float32), 'observation.images.top':np.zeros((480,640,3),dtype=np.uint8),'observation.images.wrist':np.zeros((480,640,3),dtype=np.uint8)}
    torch.manual_seed(42)
    a,old,traj=p.sample_actions_with_log_prob(obs)
    snapshot={i:{k:v.clone() for k,v in layer.items()} for i,layer in traj["prefix_kv"].items()}
    assert a.shape==(6,) and np.isfinite(a).all()
    new=p.reeval_log_prob(obs,traj)
    print('LOG_PROBS',old.item(),new.item(),flush=True)
    torch.testing.assert_close(new.detach(),old,atol=1e-3,rtol=1e-4)
    (-new).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.trainable_actor_params())
    p.policy.zero_grad();p.noise_log_std_head.zero_grad()
    batched=p.reeval_log_prob_batched([traj,traj])
    print("BATCH_LOG_PROB_DELTA", (batched.detach()-old).tolist(),flush=True)
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched[0],batched[1])
    for i,layer in snapshot.items():
        for k,v in layer.items():
            torch.testing.assert_close(traj["prefix_kv"][i][k],v,atol=0,rtol=0)
    (-batched.mean()).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.trainable_actor_params())
    print('PASS sampling, single/batched re-evaluation, actor gradients',flush=True)


if __name__ == "__main__":
    main()
