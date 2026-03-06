#!/usr/bin/env python3
"""
Multi-rollout inference for SearchQA reward distribution diagnosis.

For each prompt, runs N rollouts (with temperature sampling) and records
per-rollout rewards. Output is a JSON file consumed by plot_reward_matrix.py.

Usage:
    cd /workspace/RAGEN

    # 1. Make sure retrieval server is running:
    #    bash scripts/retrieval/launch_server.sh ./search_data/prebuilt_indices 8000

    # 2. Run inference:
    CUDA_VISIBLE_DEVICES=0 python scripts/vast/run_search_inference.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --n_prompts 50 \
        --rollouts_per_prompt 8 \
        --temperature 0.7 \
        --max_turns 5 \
        --retrieval_port 8000 \
        --output logs/search_inference.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ragen.env.search.reward import SearchRewardFn

SYSTEM_PROMPT = (
    "You are a search agent answering questions by searching for information.\n"
    "Use search[your query] to find relevant documents, and finish[your answer] to submit your final answer.\n\n"
    "You should first reason step-by-step about the current situation. "
    "This reasoning process MUST be enclosed within <think> </think> tags.\n"
    "Then provide your action within <answer>...</answer> tags.\n\n"
    "Examples:\n"
    "  <think>I need to find information about Ben Platt's father.</think>"
    "<answer>search[Ben Platt father parent]</answer>\n"
    "  <think>Based on the search results, Ben Platt's father is Henry Platt.</think>"
    "<answer>finish[Henry Platt]</answer>\n"
)


def extract_action(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    for pattern in [r"(search\[.*?\])", r"(finish\[.*?\])"]:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    return response.strip()


def search_retrieval(query: str, port: int, top_k: int = 5) -> str:
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        data = resp.json()
        results = data.get("results", [])
        lines = []
        total_chars = 0
        for i, r in enumerate(results[:top_k], 1):
            content = r.get("content", "")
            if total_chars + len(content) > 4000:
                content = content[: max(0, 4000 - total_chars)]
            total_chars += len(content)
            score = r.get("score", 0.0)
            lines.append(f"[{i}] (score: {score:.4f}) {content}")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def run_episode(question, ground_truth, llm, tokenizer, sampling_params, args):
    """Run one multi-turn episode. Returns (reward, n_turns, action_types, final_answer)."""
    reward_fn = SearchRewardFn()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAvailable actions: search[<query>], finish[<answer>]"},
    ]

    action_types = []
    for turn in range(1, args.max_turns + 1):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Skip if prompt is too long for the model
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens > 4500:  # leave room for generation
            action_types.append("truncated")
            return 0.0, turn, action_types, ""
        try:
            outputs = llm.generate([prompt], sampling_params)
        except ValueError:
            # Prompt too long for max_model_len
            action_types.append("truncated")
            return 0.0, turn, action_types, ""
        response = outputs[0].outputs[0].text

        action = extract_action(response)
        if action.startswith("search[") and action.endswith("]"):
            action_types.append("search")
            query = action[7:-1]
            results = search_retrieval(query, args.retrieval_port)
            obs = f"Search results for '{query}':\n{results}\n\nAvailable actions: search[<query>], finish[<answer>]"
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs})
        elif action.startswith("finish[") and action.endswith("]"):
            action_types.append("finish")
            answer = action[7:-1]
            reward, _ = reward_fn.compute_reward(answer, ground_truth)
            return reward, turn, action_types, answer
        else:
            action_types.append("other")
            extracted = reward_fn.extract_answer_from_response(action)
            reward, _ = reward_fn.compute_reward(extracted, ground_truth)
            return reward, turn, action_types, extracted

    return 0.0, args.max_turns, action_types, ""


def main():
    parser = argparse.ArgumentParser(description="Multi-rollout inference for reward distribution diagnosis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_prompts", type=int, default=50, help="Number of prompts to evaluate")
    parser.add_argument("--rollouts_per_prompt", type=int, default=8, help="Number of rollouts per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_turns", type=int, default=5, help="Max search turns per episode")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--retrieval_port", type=int, default=8000)
    parser.add_argument("--data_path", default="data/search/val.parquet")
    parser.add_argument("--output", default="logs/search_inference.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    n_prompts = min(args.n_prompts, len(df))

    # Seed for prompt selection (shuffle to get diverse questions)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(df))[:n_prompts]

    print(f"Loading model: {args.model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        max_model_len=5000,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=300,
    )

    print(f"Running {n_prompts} prompts x {args.rollouts_per_prompt} rollouts "
          f"(temp={args.temperature}, max_turns={args.max_turns})")
    print("=" * 60)

    results = []
    t0 = time.time()

    for pi, idx in enumerate(indices):
        row = df.iloc[idx]
        question = row["question"]
        ground_truth = row["ground_truth"]

        rollout_rewards = []
        rollout_details = []

        for ri in range(args.rollouts_per_prompt):
            reward, turns, action_types, answer = run_episode(
                question, ground_truth, llm, tokenizer, sampling_params, args
            )
            rollout_rewards.append(reward)
            rollout_details.append({
                "rollout_idx": ri,
                "reward": reward,
                "turns": turns,
                "action_types": action_types,
                "answer": answer,
            })

        rewards_arr = np.array(rollout_rewards)
        rv = float(np.var(rewards_arr))
        mean_r = float(np.mean(rewards_arr))

        results.append({
            "prompt_idx": int(idx),
            "question": question,
            "ground_truth": ground_truth if isinstance(ground_truth, str) else str(ground_truth),
            "rewards": rollout_rewards,
            "reward_mean": mean_r,
            "reward_variance": rv,
            "rollouts": rollout_details,
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1}/{n_prompts}] q=\"{question[:60]}...\" "
                  f"rewards={rollout_rewards} mean={mean_r:.3f} RV={rv:.4f} "
                  f"| {elapsed:.1f}s")

    elapsed = time.time() - t0

    # Summary stats
    all_rvs = [r["reward_variance"] for r in results]
    all_means = [r["reward_mean"] for r in results]
    n_zero_rv = sum(1 for rv in all_rvs if rv == 0.0)
    n_all_correct = sum(1 for r in results if all(rw == 1.0 for rw in r["rewards"]))
    n_all_wrong = sum(1 for r in results if all(rw == 0.0 for rw in r["rewards"]))
    n_mixed = n_prompts - n_all_correct - n_all_wrong

    print(f"\n{'=' * 60}")
    print(f"INFERENCE SUMMARY ({n_prompts} prompts x {args.rollouts_per_prompt} rollouts, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Mean reward across all:  {np.mean(all_means):.4f}")
    print(f"Mean RV:                 {np.mean(all_rvs):.4f}")
    print(f"Prompts with RV=0:       {n_zero_rv}/{n_prompts} ({n_zero_rv/n_prompts*100:.1f}%)")
    print(f"  - All correct (easy):  {n_all_correct}")
    print(f"  - All wrong (hard):    {n_all_wrong}")
    print(f"  - Mixed (learnable):   {n_mixed}")
    print()

    if n_mixed / max(n_prompts, 1) < 0.2:
        print("WARNING: <20% prompts have mixed rewards. RL signal may be weak.")
        print("  -> Check prompt format, retrieval server, and model capability.")
    else:
        print(f"OK: {n_mixed/n_prompts*100:.0f}% prompts have mixed rewards. RL should have signal.")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "model": args.model,
            "n_prompts": n_prompts,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "data_path": args.data_path,
            "seed": args.seed,
        },
        "summary": {
            "mean_reward": float(np.mean(all_means)),
            "mean_rv": float(np.mean(all_rvs)),
            "n_all_correct": n_all_correct,
            "n_all_wrong": n_all_wrong,
            "n_mixed": n_mixed,
            "elapsed_seconds": elapsed,
        },
        "prompts": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print(f"Next step: python scripts/vast/plot_reward_matrix.py --input {output_path}")


if __name__ == "__main__":
    main()
