#!/usr/bin/env python3
"""Read public Codabench ranks and persist a sanitized, timestamped snapshot.

Uses no credentials, submission endpoint, detailed-result link, or private data.
Only owner names, public numeric scores, entry dates and ranking settings survive.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"results/optimization/leaderboard_review"
BOARD="https://www.codabench.org/api/leaderboards/19740/"
COMPETITION="https://www.codabench.org/api/competitions/17739/"


def fetch(url):
    response=requests.get(url,timeout=30);response.raise_for_status();return response.json()


def main():
    board,competition=fetch(BOARD),fetch(COMPETITION)
    stamp=datetime.now(timezone.utc)
    rows=[]
    for entry in board["submissions"]:
        scores={item["column_key"]:float(item["score"]) for item in entry.get("scores",[]) if item.get("score") is not None}
        rows.append({"submission_id":entry["id"],"owner":entry.get("owner"),"created_utc":entry.get("created_when"),"scores":scores})
    config=next(row for row in competition["leaderboards"] if row["id"]==19740)
    summary={"observed_utc":stamp.isoformat(),"observed_local":stamp.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "source":BOARD,"competition_source":COMPETITION,"entries":len(rows),"distinct_owners":len({row["owner"] for row in rows}),
        "submission_rule":config.get("submission_rule"),"primary_index":config.get("primary_index"),
        "phases":[{key:phase.get(key) for key in ("id","name","start","end","description","max_submissions_per_day","max_submissions_per_person")} for phase in competition["phases"]],
        "rank_definition":"1 + number of displayed entries having a strictly larger value; ties explicitly reported",
        "rows":rows}
    ours=[row for row in rows if row["owner"]=="shansiming"]
    summary["our_visible_entries"]=ours
    for column in ("d3d_S","g_ratio"):
        ordered=sorted(rows,key=lambda row:(-row["scores"].get(column,0),row["submission_id"]))
        summary[column+"_top10"]=ordered[:10]
        summary[column+"_user_ranks"]=[{"submission_id":row["submission_id"],"score":row["scores"].get(column,0),
            "rank":1+sum(item["scores"].get(column,0)>row["scores"].get(column,0) for item in rows),
            "tie_size":sum(item["scores"].get(column,0)==row["scores"].get(column,0) for item in rows)} for row in ours]
    best_d3d=summary["d3d_S_top10"][0]["scores"]["d3d_S"]
    best_g=summary["g_ratio_top10"][0]["scores"]["g_ratio"]
    summary["metric_feasibility"]={"score_upper_bound":1.,"eligibility_d3d_S":.85,"g_ratio_is_clipped_at_one":False,
        "current_leading_d3d_S":best_d3d,"current_leading_g_ratio":best_g,
        "maximum_g_ratio_at_leading_d3d_S":1/best_d3d,
        "mast_S_needed_at_leading_d3d_S_to_tie_leading_G":best_g*best_d3d,
        "maximum_d3d_S_for_leading_G_even_with_perfect_MAST":1/best_g,
        "one_entry_can_beat_both_current_leaders":1/best_d3d>best_g}
    OUTPUT.mkdir(exist_ok=True)
    text=json.dumps(summary,indent=2,allow_nan=False)+"\n"
    (OUTPUT/"public_snapshot.json").write_text(text)
    (OUTPUT/f"public_snapshot_{stamp.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(text)
    print(json.dumps({"observed_utc":summary["observed_utc"],"entries":len(rows),"submission_rule":summary["submission_rule"],
        "our_ch1_ranks":summary["d3d_S_user_ranks"],"our_ch2_ranks":summary["g_ratio_user_ranks"],
        "ch1_leader":summary["d3d_S_top10"][0],"ch2_leader":summary["g_ratio_top10"][0]},indent=2))


if __name__=="__main__":main()
