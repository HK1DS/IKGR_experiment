'''
RAG expand related intents (from fixed intent vocab)
'''

import yaml, numpy as np
import pandas as pd
from tqdm import tqdm
from ikgr_core.utils import load_json, save_json, read_csv, write_csv, ensure_dir
from ikgr_core.rag import IntentEncoderIndex, knn_strings
from ikgr_core.llm_client import LocalLLM
import re, ast, json

def _safe_eval_list(s):
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    
    # Strip markdown block quotes
    s = re.sub(r"^```(?:python|json)?", "", s, flags=re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE)
    s = s.strip()
    
    # Extract everything inside [ and ]
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if match:
        s_content = match.group(0)
    else:
        s_content = s
        
    try:
        val = ast.literal_eval(s_content)
        if isinstance(val, list):
            return val
    except:
        pass
        
    try:
        val = json.loads(s_content)
        if isinstance(val, list):
            return val
    except:
        pass
        
    # Regex fallback for integers or single/double quoted items
    try:
        # Check if they are indices (integers)
        nums = [int(x) for x in re.findall(r"\d+", s_content)]
        if nums:
            return nums
    except:
        pass
        
    return []

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    paths = cfg["paths"]
    rag_cfg = cfg["rag"]
    llm_cfg = cfg["llm"]

    ensure_dir(paths["workdir"])
    df = read_csv(paths["step1_output"]).fillna("")

    # 1) Freeze vocabulary from step1 exact intents
    def _parse_exact_list(s):
        val = _safe_eval_list(s)
        return [str(x) for x in val if isinstance(x, str)]

    vocab = set()
    for col in ["user_intents_exact", "item_intents_exact"]:
        for v in df[col].dropna():
            vocab.update(_parse_exact_list(v))
    vocab = sorted(list(vocab))
    save_json(vocab, rag_cfg["vocab_json"])

    # 2) Encode vocab + build Index
    enc = IntentEncoderIndex(rag_cfg["encoder"])
    emb = enc.encode(vocab)
    np.save(rag_cfg["encoding_npy"], emb)
    enc.build_ann(emb, emb.shape[1], rag_cfg["annoy_trees"], rag_cfg["annoy_index"])

    ann = enc.load_ann(emb.shape[1], rag_cfg["annoy_index"])
    llm = LocalLLM(**llm_cfg)
    sys_prompt = "You are a helpful assistant that returns ONLY JSON lists of integers."

    # 3) For each row, expand related intents for user & item via RAG + LLM selection
    import os, time, requests
    cache_path = os.path.join(paths["workdir"], "step2_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded cache from {cache_path} (users={len(cache.get('user', {}))}, items={len(cache.get('item', {}))})")
    else:
        cache = {"user": {}, "item": {}}

    def save_cache():
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def chat_with_retry(sys_prompt, prompt, max_retries=10, initial_delay=5):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return llm.chat(sys_prompt, prompt).strip()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                if status == 429:
                    print(f"\n[Rate Limit 429] Sleeping for {delay} seconds before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    print(f"\n[HTTP Error {status}] Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
            except Exception as e:
                print(f"\n[Error] {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return llm.chat(sys_prompt, prompt).strip()

    rel_user, rel_item = [], []
    p_rel = open("prompts/step2_related.txt", "r", encoding="utf-8").read()
    save_counter = 0

    for _, r in tqdm(df.iterrows(), total=len(df)):
        u_prof, i_prof = r.get("user_profile", ""), r.get("item_profile", "")
        u_exact = _parse_exact_list(r.get("user_intents_exact", "[]"))
        i_exact = _parse_exact_list(r.get("item_intents_exact", "[]"))

        # user
        if u_prof:
            if u_prof in cache["user"]:
                u_rel = cache["user"][u_prof]
            else:
                q_emb = enc.encode([u_prof])[0]
                options = knn_strings(ann, q_emb, vocab, cfg["rag"]["knn_k"])
                options_filtered = [o for o in options if o not in u_exact]
                
                options_text = "\n".join([f"{idx}: {opt}" for idx, opt in enumerate(options_filtered)])
                prompt = p_rel.replace("{PROFILE}", u_prof).replace("{OPTIONS}", options_text)
                
                ans = chat_with_retry(sys_prompt, prompt)
                selected_indices = _safe_eval_list(ans)
                
                u_rel = []
                for idx in selected_indices:
                    try:
                        idx_int = int(idx)
                        if 0 <= idx_int < len(options_filtered):
                            u_rel.append(options_filtered[idx_int])
                    except:
                        pass
                
                cache["user"][u_prof] = u_rel
                save_counter += 1
        else:
            u_rel = []

        # item
        if i_prof:
            if i_prof in cache["item"]:
                i_rel = cache["item"][i_prof]
            else:
                q_emb = enc.encode([i_prof])[0]
                options = knn_strings(ann, q_emb, vocab, cfg["rag"]["knn_k"])
                options_filtered = [o for o in options if o not in i_exact]
                
                options_text = "\n".join([f"{idx}: {opt}" for idx, opt in enumerate(options_filtered)])
                prompt = p_rel.replace("{PROFILE}", i_prof).replace("{OPTIONS}", options_text)
                
                ans = chat_with_retry(sys_prompt, prompt)
                selected_indices = _safe_eval_list(ans)
                
                i_rel = []
                for idx in selected_indices:
                    try:
                        idx_int = int(idx)
                        if 0 <= idx_int < len(options_filtered):
                            i_rel.append(options_filtered[idx_int])
                    except:
                        pass
                
                cache["item"][i_prof] = i_rel
                save_counter += 1
        else:
            i_rel = []

        if save_counter >= 20:
            save_cache()
            save_counter = 0

        rel_user.append(str(u_rel))
        rel_item.append(str(i_rel))

    save_cache()

    df["user_intents_related"] = rel_user
    df["item_intents_related"] = rel_item
    write_csv(df, paths["step2_output"])
    print(f"[step2] saved: {paths['step2_output']}")

if __name__ == "__main__":
    main()
