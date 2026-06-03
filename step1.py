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

    user_out, item_out = [], []
    save_counter = 0
    for _, r in tqdm(df.iterrows(), total=len(df)):
        uprof = str(r.get("user_profile", "")).strip()
        iprof = str(r.get("item_profile", "")).strip()

        # user intents
        if uprof:
            if uprof in cache["user"]:
                u_ans = cache["user"][uprof]
            else:
                up = p_user.replace("{PROFILE}", uprof)
                u_ans = chat_with_retry(sys_prompt, up)
                cache["user"][uprof] = u_ans
                save_counter += 1
        else:
            u_ans = "[]"

        # item intents
        if iprof:
            if iprof in cache["item"]:
                i_ans = cache["item"][iprof]
            else:
                ip = p_item.replace("{PROFILE}", iprof)
                i_ans = chat_with_retry(sys_prompt, ip)
                cache["item"][iprof] = i_ans
                save_counter += 1
        else:
            i_ans = "[]"

        # Save cache every 20 additions
        if save_counter >= 20:
            save_cache()
            save_counter = 0

        user_out.append(u_ans if u_ans.startswith("[") else "[]")
        item_out.append(i_ans if i_ans.startswith("[") else "[]")

    save_cache()
    df["user_intents_exact"] = user_out
    df["item_intents_exact"] = item_out
    write_csv(df, out_csv)
    print(f"[step1] saved: {out_csv}")

if __name__ == "__main__":
    main()
