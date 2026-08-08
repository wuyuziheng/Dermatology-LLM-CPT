#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pubmed_parallel_flesch.py - 多设备并行版 (适配 a2.py 架构)
任务：读取 JSONL -> 计算 Flesch Grade Level -> 注入 stats -> 输出新 JSONL
"""
import os
import json
import re
import spacy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

# ================= 配置区 =================
INPUT_JSONL = "pubmed-abstract.jsonl"   # 任务一输出的文件
OUTPUT_JSONL = "a_text_dataset.jsonl" # 并行处理结果
NUM_DEVICES = None                              # None 表示自动检测，也可手动指定 (如 4)
# ==========================================

# ================= 工具函数 =================
def count_syllables(word: str) -> int:
    """启发式音节计数"""
    word = word.lower()
    if len(word) <= 3:
        return 1
    if word.endswith('e'):
        word = word[:-1]
    vowels, count, prev = "aeiouy", 0, False
    for char in word:
        is_v = char in vowels
        if is_v and not prev:
            count += 1
        prev = is_v
    if len(word) >= 2 and word.endswith('le') and word[-3] not in vowels:
        count += 1
    return max(count, 1)

def calculate_flesch_grade_level(text: str, nlp) -> float:
    """计算 Flesch-Kincaid Grade Level (接收局部 nlp 实例)"""
    doc = nlp(text)
    num_sentences = max(len(list(doc.sents)), 1)
    words = re.findall(r'\b[\w\'\-]+\b', text)
    num_words = len(words)
    if num_words == 0:
        return 0.0
    total_syl = sum(count_syllables(w) for w in words)
    avg_sl = num_words / num_sentences
    avg_sw = total_syl / num_words
    return round(0.39 * avg_sl + 11.8 * avg_sw - 15.59, 2)

# ================= 并行工作进程 =================
def worker_process_chunk(lines_chunk: list, device_id: int, tmp_output: str) -> int:
    """
    单设备工作进程：严格隔离上下文，独立初始化，流式写入临时文件
    """
    # 1. 设备隔离 (完全对齐 a2.py 思想)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # 若后续替换为 GPU 加速模型 (如 cuBERT/transformers)，取消下方注释即可：
    # import torch
    # torch.cuda.set_device(0)

    # 2. 进程内独立初始化 spacy (避免跨进程 Pickle 序列化冲突)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    records_written = 0
    with open(tmp_output, 'w', encoding='utf-8') as f_out:
        # 3. 流式处理分配的行块
        for line in tqdm(lines_chunk, desc=f"Device {device_id}", leave=False, position=device_id):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 4. 文本清洗与计算
            raw_text = record.get("text", "")
            text = raw_text.replace("\n", " ").replace("\r", " ").strip()
            flesch_score = calculate_flesch_grade_level(text, nlp)

            # 5. 组装目标格式
            stats = record.get("stats", {})
            stats["flesch_grade_level"] = flesch_score

            new_record = {
                "stats": stats,
                "meta": record.get("meta", {}),
                "text": text,
                "simhash": record.get("simhash", 0)
            }
            f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            records_written += 1

    return records_written

# ================= 主并行调度函数 =================
def process_jsonl_parallel(input_jsonl: str, output_jsonl: str, num_devices: int = None) -> None:
    """
    并行调度核心：分片 -> 分发 -> 等待 -> 合并 (原子写入)
    """
    # 自动检测可用设备数 (优先 GPU，无 GPU 则回退 CPU 逻辑核心数)
    if num_devices is None:
        try:
            import torch
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                print(f"检测到 {num_devices} 张 GPU，将启动 CUDA 并行进程...")
            else:
                num_devices = mp.cpu_count()
                print(f"未检测到 GPU，回退至 {num_devices} 个 CPU 核心并行...")
        except ImportError:
            num_devices = mp.cpu_count()
            print(f"使用 {num_devices} 个 CPU 核心并行...")

    # 1. 读取并切分数据 (内存安全：仅加载行指针)
    print(f"加载并切分数据: {input_jsonl}")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        all_lines = [line for line in f if line.strip()]

    if not all_lines:
        print("输入文件为空或无有效数据")
        return

    # 轮询分片 (保证各进程负载均衡)
    chunks = [all_lines[i::num_devices] for i in range(num_devices)]
    tmp_files = [f"{output_jsonl}.gpu{i}.tmp" for i in range(num_devices)]

    # 2. 强制使用 spawn 启动方式 (避免 CUDA/spacy 全局状态污染)
    mp.set_start_method('spawn', force=True)
    total_records = 0

    with ProcessPoolExecutor(max_workers=num_devices) as executor:
        futures = {}
        for dev_id, chunk in enumerate(chunks):
            if chunk:  # 仅提交非空任务
                fut = executor.submit(worker_process_chunk, chunk, dev_id, tmp_files[dev_id])
                futures[fut] = dev_id

        # 3. 收集结果
        for future in as_completed(futures):
            dev_id = futures[future]
            try:
                count = future.result()
                total_records += count
            except Exception as e:
                print(f"Device {dev_id} 进程异常: {e}")

    # 4. 原子合并临时文件 (逐行拷贝，避免大文件 OOM)
    print("正在合并临时文件...")
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for tmp_path in tmp_files:
            if os.path.exists(tmp_path):
                with open(tmp_path, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                os.remove(tmp_path)  # 清理中间文件

    print(f"并行处理完成 | 成功写入: {total_records} 条 | 输出: {output_jsonl}")

if __name__ == "__main__":

    
    process_jsonl_parallel(INPUT_JSONL, OUTPUT_JSONL, NUM_DEVICES)