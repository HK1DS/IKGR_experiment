'''
Extract exact intents
'''

import yaml, os
import pandas as pd
from tqdm import tqdm
from ikgr_core.llm_client import LocalLLM
from ikgr_core.utils import read_csv, write_csv, ensure_dir

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    in_csv = cfg["paths"]["input_csv"]
    out_csv = cfg["paths"]["step1_output"]
    work = cfg["paths"]["workdir"]
    ensure_dir(work)

    llm_cfg = cfg["llm"]
    llm = LocalLLM(**llm_cfg)

    df = read_csv(in_csv).fillna("")
    # expected columns: user_id,user_profile,item_id,item_profile
    sys_prompt = "You are a helpful assistant that returns ONLY Python lists."

    p_user = load_prompt("prompts/step1_intents.txt")
    p_item = p_user  # same template

    import json, time, requests
    cache_path = os.path.join(work, "step1_cache.json")
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

    # Extract unique profiles
    unique_users = df[df["user_profile"] != ""]["user_profile"].unique()
    unique_items = df[df["item_profile"] != ""]["item_profile"].unique()

    print(f"Unique Users: {len(unique_users)}, Unique Items: {len(unique_items)}")

    save_counter = 0
    # Process unique user profiles
    for uprof in tqdm(unique_users, desc="Extracting User Intents"):
        uprof = str(uprof).strip()
        if uprof not in cache["user"]:
            up = p_user.replace("{PROFILE}", uprof)
            u_ans = chat_with_retry(sys_prompt, up)
            cache["user"][uprof] = u_ans if u_ans.startswith("[") else "[]"
            save_counter += 1
            if save_counter >= 20:
                save_cache()
                save_counter = 0

    # Process unique item profiles
    for iprof in tqdm(unique_items, desc="Extracting Item Intents"):
        iprof = str(iprof).strip()
        if iprof not in cache["item"]:
            ip = p_item.replace("{PROFILE}", iprof)
            i_ans = chat_with_retry(sys_prompt, ip)
            cache["item"][iprof] = i_ans if i_ans.startswith("[") else "[]"
            save_counter += 1
            if save_counter >= 20:
                save_cache()
                save_counter = 0

    save_cache()

    # Fast pandas mapping to populate the dataframe
    print("Mapping intents back to main dataframe...")
    df["user_intents_exact"] = df["user_profile"].map(lambda x: cache["user"].get(str(x).strip(), "[]"))
    df["item_intents_exact"] = df["item_profile"].map(lambda x: cache["item"].get(str(x).strip(), "[]"))
    
    write_csv(df, out_csv)
    print(f"[step1] saved: {out_csv}")

if __name__ == "__main__":
    main()
