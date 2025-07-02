import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sleeper_wrapper import League, Players, Stats
from requests import HTTPError
from collections import defaultdict
from datetime import datetime

# ----------- CONFIGURATION -----------------
LEAGUE_IDS_BY_YEAR = {
    2023: '947624080541310976',
    2024: '1045743296628244480',
    2025: '1177710603576954880',
}

LEAGUE_IDS_BY_YEAR_PLAYOFFS = {
    2023: '947624080541310976',
    2024: '1045743296628244480',
}

st.set_page_config(layout="wide", page_title="Fantasy League Dashboard")

# ----------- STYLING ------------------------
# Custom CSS for Poppins, titles, table color/padding
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, .stApp {
        font-family: 'Poppins', Arial, sans-serif !important;
        background-color: #001f3f !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', Arial, sans-serif !important;
        color: white !important;
    }

    /* Table titles */
    .stMarkdown > h1, .stMarkdown > h2, .stMarkdown > h3 {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: white !important;
        margin-bottom: 12px !important;
    }

    /* DataFrame table styling */
    .stDataFrame, .stTable {
        background-color: #10213C !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 16px !important;
        color: white !important;
        font-family: 'Poppins', Arial, sans-serif !important;
    }
    /* Table header */
    thead tr th {
        background-color: #10213C !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 10px 6px !important;
    }
    /* Table rows */
    tbody tr td {
        background-color: #10213C !important;
        color: white !important;
        padding: 8px 6px !important;
        border-bottom: 1px solid #29365B !important;
    }

    /* Remove annoying Streamlit line under tables */
    .stDataFrame > div {
        border: none !important;
    }

    /* Remove left gap/padding for better stretch */
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# ----------- HELPERS (no lambdas in cache) ------------
def reg_stat_template():
    return {'wins': 0, 'losses': 0, 'pf': 0.0, 'pa': 0.0, 'gp': 0, 'display': '', 'team': ''}

def playoff_stat_template():
    return {'wins': 0, 'losses': 0, 'pf': 0.0, 'gp': 0, 'display': '', 'team': '', 'made_playoffs': False}

@st.cache_data(show_spinner=True)
def track_regular_season_only_stats(ids_by_year):
    stats = defaultdict(reg_stat_template)
    for year, lid in ids_by_year.items():
        lg = League(lid)
        try:
            rosters = lg.get_rosters()
            users = lg.get_users()
        except:
            continue
        r2u = {r['roster_id']: r['owner_id'] for r in rosters}
        uid2n = {u['user_id']: u['display_name'] for u in users}
        uid2t = {u['user_id']: u.get('metadata', {}).get('team_name', f'Unknown ({year})') for u in users}
        for uid in uid2n:
            stats[uid]['display'] = uid2n[uid]
            stats[uid]['team'] = uid2t[uid]
        for wk in range(1, 14):
            try:
                matchups = lg.get_matchups(wk)
            except:
                continue
            by_id = defaultdict(list)
            for m in matchups:
                by_id[m['matchup_id']].append(m)
            for pair in by_id.values():
                if len(pair) != 2:
                    continue
                m1, m2 = pair
                u1, u2 = r2u.get(m1['roster_id']), r2u.get(m2['roster_id'])
                if not u1 or not u2:
                    continue
                p1, p2 = m1.get('points', 0), m2.get('points', 0)
                stats[u1]['pf'] += p1
                stats[u1]['pa'] += p2
                stats[u1]['gp'] += 1
                stats[u2]['pf'] += p2
                stats[u2]['pa'] += p1
                stats[u2]['gp'] += 1
                if p1 > p2:
                    stats[u1]['wins'] += 1
                    stats[u2]['losses'] += 1
                elif p2 > p1:
                    stats[u2]['wins'] += 1
                    stats[u1]['losses'] += 1
    return stats

@st.cache_data(show_spinner=True)
def track_playoff_only_stats(ids_by_year):
    stats = defaultdict(playoff_stat_template)
    for year, lid in ids_by_year.items():
        lg = League(lid)
        try:
            league_info = lg.get_league()
            start_wk = int(league_info['settings'].get('playoff_week_start', 15))
            rosters, users = lg.get_rosters(), lg.get_users()
        except:
            continue
        sorted_ro = sorted(rosters, key=lambda r: r['settings'].get('wins', 0), reverse=True)[:6]
        r2u = {r['roster_id']: r['owner_id'] for r in rosters}
        uid2n = {u['user_id']: u['display_name'] for u in users}
        uid2t = {u['user_id']: u.get('metadata', {}).get('team_name', f'Unknown ({year})') for u in users}
        for r in sorted_ro:
            uid = r['owner_id']
            stats[uid]['made_playoffs'] = True
            stats[uid]['display'] = uid2n[uid]
            stats[uid]['team'] = uid2t[uid]
        alive = {r['roster_id'] for r in sorted_ro}
        for wk in range(start_wk, 18):
            try:
                matchups = lg.get_matchups(wk)
            except:
                continue
            by_id = defaultdict(list)
            for m in matchups:
                by_id[m['matchup_id']].append(m)
            next_alive = set()
            for pair in by_id.values():
                if len(pair) != 2:
                    continue
                m1, m2 = pair
                r1, r2 = m1['roster_id'], m2['roster_id']
                if r1 not in alive and r2 not in alive:
                    continue
                u1, u2 = r2u[r1], r2u[r2]
                p1, p2 = m1.get('points', 0), m2.get('points', 0)
                stats[u1]['pf'] += p1
                stats[u2]['pf'] += p2
                stats[u1]['gp'] += 1
                stats[u2]['gp'] += 1
                if p1 > p2:
                    stats[u1]['wins'] += 1
                    stats[u2]['losses'] += 1
                    next_alive.add(r1)
                elif p2 > p1:
                    stats[u2]['wins'] += 1
                    stats[u1]['losses'] += 1
                    next_alive.add(r2)
            alive = next_alive
    return stats

def get_combined_wins(reg_stats, playoff_stats):
    combined = {}
    for uid, s in reg_stats.items():
        combined[uid] = dict(s)
    for uid, s in playoff_stats.items():
        if uid not in combined:
            combined[uid] = dict(s)
        else:
            for k in ('wins', 'losses', 'pf', 'gp'):
                combined[uid][k] += s[k]
    data = []
    for s in combined.values():
        avg_pf = int(round(s['pf'] / s['gp'], 0)) if s['gp'] > 0 else 0
        win_pct = round(100*s['wins']/s['gp'],1) if s['gp'] > 0 else 0.0
        data.append({
            'Manager': s['display'],
            'Team': s['team'],
            'Wins': s['wins'],
            'Losses': s['losses'],
            'Win %': round(win_pct),
            'PF': int(round(s['pf'], 0)),
            'PA': int(round(s['pa'], 0)),
            'GP': s['gp'],
            'Avg PF': avg_pf
        })
    df = pd.DataFrame(data)
    df = df.sort_values(['Wins', 'PF'], ascending=[False, False]).reset_index(drop=True)
    df.index += 1
    df.index.name = "Pos"
    return df

def get_playoff_wins_table(playoff_stats):
    data = []
    for s in playoff_stats.values():
        # Only include teams that made playoffs AND have at least one appearance
        # (gp > 0) or recorded wins/losses to avoid empty entries
        if s['made_playoffs'] and (s['gp'] > 0 or s['wins'] > 0 or s['losses'] > 0):
            avg_pf = int(round(s['pf'] / s['gp'], 0)) if s['gp'] > 0 else 0
            data.append({
                'Manager': s['display'],
                'Team': s['team'],
                'Wins': s['wins'],
                'Losses': s['losses'],
                'Games Played': s['gp'],
                'Avg PF': avg_pf
            })
    df = pd.DataFrame(data)
    df = df.sort_values(['Wins', 'Avg PF'], ascending=[False, False]).reset_index(drop=True)
    df.index += 1
    df.index.name = "Pos"
    return df


def plot_quadrant(reg_stats, playoff_stats, title="Wins vs Avg PF"):
    labels, wins, avg_pf = [], [], []
    for uid in reg_stats:
        r, p = reg_stats[uid], playoff_stats.get(uid, {})
        labels.append(r['display'])
        total_wins = r['wins'] + p.get('wins', 0)
        total_pf = r['pf'] + p.get('pf', 0.0)
        total_gp = r['gp'] + p.get('gp', 0)
        wins.append(total_wins)
        avg_pf.append(int(round(total_pf / total_gp, 0)) if total_gp > 0 else 0)

    avg_w = sum(wins) / len(wins)
    avg_pf_v = sum(avg_pf) / len(avg_pf)

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#d8e2ed')  # background colour
    ax.set_facecolor('#d8e2ed')

    # Scatter points
    ax.scatter(avg_pf, wins, s=100, color='#001f3f')

    # Grid lines
    ax.axhline(avg_w, color='#001f3f', linestyle='--')
    ax.axvline(avg_pf_v, color='#001f3f', linestyle='--')

    # Quadrant labels
    ax.text(avg_pf_v+0.5, avg_w+1, 'Elite', fontsize=12, fontweight='bold', color='#001f3f')
    ax.text(avg_pf_v-7,   avg_w+1, 'Overachievers', fontsize=12, fontweight='bold', color='#001f3f')
    ax.text(avg_pf_v+0.5, avg_w-3, 'Underperformers', fontsize=12, fontweight='bold', color='#001f3f')
    ax.text(avg_pf_v-7,   avg_w-3, 'Struggling', fontsize=12, fontweight='bold', color='#001f3f')

    # Data point labels
    texts = []
    for i, (x, y) in enumerate(zip(avg_pf, wins)):
        texts.append(ax.text(x, y, labels[i], fontsize=9, color='#001f3f', fontname='Poppins'))

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color='#001f3f', lw=0.5),
        expand_points=(1.5, 1.5)
    )

    ax.set_xlabel("Average Points For", color='#001f3f', fontname='Poppins')
    ax.set_ylabel("Total Wins (Reg + Playoffs)", color='#001f3f', fontname='Poppins')
    ax.set_title(title, color='#001f3f', fontname='Poppins')
    ax.tick_params(axis='x', colors='#001f3f')
    ax.tick_params(axis='y', colors='#001f3f')
    ax.grid(True, color='#001f3f')
    fig.tight_layout()
    st.pyplot(fig)

@st.cache_data(show_spinner=True)
def get_head_to_head_heatmap():
    combined = defaultdict(lambda: defaultdict(lambda: {'wins':0,'losses':0}))
    team_names = set()
    for season, lid in LEAGUE_IDS_BY_YEAR.items():
        lg = League(lid)
        rs, us = lg.get_rosters(), lg.get_users()
        r2u = {r['roster_id']: r['owner_id'] for r in rs}
        u2n = {u['user_id']: u['display_name'] for u in us}
        for wk in range(1, 14):
            try:
                matchups = lg.get_matchups(wk)
            except:
                continue
            byid = defaultdict(list)
            for g in matchups:
                byid[g['matchup_id']].append(g)
            for pair in byid.values():
                if len(pair) != 2:
                    continue
                a, b = pair
                u1 = r2u[a['roster_id']]; u2 = r2u[b['roster_id']]
                n1 = u2n[u1]; n2 = u2n[u2]
                team_names.update([n1, n2])
                p1, p2 = a.get('points',0), b.get('points',0)
                if p1 > p2:
                    combined[n1][n2]['wins']   += 1
                    combined[n2][n1]['losses'] += 1
                elif p2 > p1:
                    combined[n2][n1]['wins']   += 1
                    combined[n1][n2]['losses'] += 1
    teams = sorted(team_names)
    import numpy as np
    n = len(teams)
    mat = np.zeros((n,n))
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if t1 == t2:
                mat[i,j] = np.nan
            else:
                mat[i,j] = combined[t1][t2]['wins']

    # PLOTTING BLOCK
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor('#d8e2ed')  # Figure background
    ax.set_facecolor('#d8e2ed')         # Axes background

    cax = ax.imshow(mat, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(teams, rotation=45, ha='right', fontsize=9, color='#001f3f', fontname='Poppins')
    ax.set_yticklabels(teams, fontsize=9, color='#001f3f', fontname='Poppins')

    # Annotate cells with win values
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(mat[i,j]):
                ax.text(j, i, int(mat[i,j]), ha='center', va='center', fontsize=9, color='#001f3f', fontname='Poppins')

    cbar = plt.colorbar(cax, ax=ax, label='Wins vs Opponent')
    cbar.ax.yaxis.label.set_color('#001f3f')
    cbar.ax.tick_params(colors='#001f3f')
    ax.set_title("All-Time Regular Season Head-to-Head Wins", color='#001f3f', fontname='Poppins')
    ax.grid(False)
    fig.tight_layout()
    st.pyplot(fig)

    return teams, mat


# --------- TRADES LOGIC -------------
@st.cache_data(show_spinner=True)
def lookup_player_names():
    raw = Players().get_all_players()
    return {
        pid: pdata.get('full_name') 
             or f"{pdata.get('first_name','')} {pdata.get('last_name','')}".strip()
             or pid
        for pid, pdata in raw.items()
    }

def safe_tx(lg, wk):
    try:
        txs = lg.get_transactions(wk)
        if isinstance(txs, HTTPError):
            raise txs
        return txs if isinstance(txs, list) else []
    except:
        return []

@st.cache_data(show_spinner=True)
def build_detailed_ledger(ids_by_year):
    player_names = lookup_player_names()
    name_to_pid  = {v:k for k,v in player_names.items()}
    stats_api    = Stats()
    week_cache   = {}

    def get_week(season, wk):
        key = (season, wk)
        if key not in week_cache:
            lines = stats_api.get_week_stats('regular', season, wk)
            if isinstance(lines, dict):
                week_cache[key] = lines
            else:
                week_cache[key] = { str(l['player_id']): l for l in lines }
        return week_cache[key]

    trades_made = defaultdict(lambda: defaultdict(int))
    for season, lid in ids_by_year.items():
        lg = League(lid)
        rs = lg.get_rosters()
        r2u = {r['roster_id']: r['owner_id'] for r in rs}
        for wk in range(1, 18):
            for tx in safe_tx(lg, wk):
                if tx.get('type')!='trade': continue
                for rid in (tx.get('roster_ids') or []):
                    trades_made[season][r2u.get(rid)] += 1

    rows = []
    for season, lid in ids_by_year.items():
        lg = League(lid)
        rs = lg.get_rosters(); us = lg.get_users()
        r2u = {r['roster_id']: r['owner_id'] for r in rs}
        uid2n = {u['user_id']:u['display_name'] for u in us}
        for wk in range(1,18):
            for tx in safe_tx(lg, wk):
                if tx.get('type')!='trade': continue
                adds       = tx.get('adds') or {}
                drops      = tx.get('drops') or {}
                picks      = tx.get('draft_picks') or []
                roster_ids = tx.get('roster_ids') or []
                if len(roster_ids)<2: continue
                dt = datetime.utcfromtimestamp(int(tx['status_updated'])/1000)
                date_str = dt.strftime('%Y-%m-%d')
                ledger = { rid:{'in_p':set(),'in_pk':[],'out_p':set()} for rid in roster_ids }
                for pid,rid in adds.items():
                    if rid in ledger:
                        ledger[rid]['in_p'].add(player_names.get(pid,pid))
                for pid,rid in drops.items():
                    if rid in ledger:
                        ledger[rid]['out_p'].add(player_names.get(pid,pid))
                for pk in picks:
                    rid = pk.get('roster_id')
                    if rid in ledger:
                        desc = f"{pk.get('season')} R{pk.get('round')}"
                        ledger[rid]['in_pk'].append(desc)
                for i in range(len(roster_ids)):
                    for j in range(i+1, len(roster_ids)):
                        a, b = roster_ids[i], roster_ids[j]
                        ua, ub = r2u[a], r2u[b]
                        name_a, name_b = uid2n.get(ua,'?'), uid2n.get(ub,'?')
                        in_p_a   = sorted(ledger[a]['in_p'])
                        out_p_a  = sorted(ledger[a]['out_p'])
                        in_pk_a  = ledger[a]['in_pk']
                        out_pk_a = ledger[b]['in_pk']
                        rows.append({
                            'Date'         : date_str,
                            'Season'       : season,
                            'Week'         : wk,
                            'Manager A'    : name_a,
                            'Incoming A'   : "; ".join(in_p_a),
                            'Picks In A'   : "; ".join(in_pk_a),
                            'Manager B'    : name_b,
                            'Incoming B'   : "; ".join(sorted(ledger[b]['in_p'])),
                            'Picks In B'   : "; ".join(ledger[b]['in_pk']),
                        })
    return pd.DataFrame(rows)


# ---------- MAIN DASHBOARD ------------------
reg_stats = track_regular_season_only_stats(LEAGUE_IDS_BY_YEAR)
playoff_stats = track_playoff_only_stats(LEAGUE_IDS_BY_YEAR)

# Most Winningest Coaches
st.header("MWG - Most Winningest Coach")
df_combined = get_combined_wins(reg_stats, playoff_stats)
st.table(df_combined)

# Teams who made playoffs every year
st.header("Teams Who Made Playoffs Every Year")
total_seasons = len(LEAGUE_IDS_BY_YEAR_PLAYOFFS)
playoff_appearances = defaultdict(list)
for year, lid in LEAGUE_IDS_BY_YEAR_PLAYOFFS.items():
    po = track_playoff_only_stats({year: lid})
    for uid, s in po.items():
        if s['made_playoffs']:
            playoff_appearances[uid].append((year, s['team']))
teams_made_playoffs_all = []
for uid, years in playoff_appearances.items():
    if len(years) == total_seasons:
        teams_made_playoffs_all.append(years[-1][1])
if teams_made_playoffs_all:
    st.markdown(
        "<div style='text-align:center; margin-top:16px; margin-bottom:16px;'>"
        + "<br>".join(
            [f"<span style='font-family:Poppins,sans-serif; font-size:22px; font-weight:bold; color:#d8e2ed;'>{team}</span>" 
             for team in teams_made_playoffs_all]
          )
        + "</div>", 
        unsafe_allow_html=True
    )
else:
    st.write("No team has made playoffs every year.")

# Quadrant Plot with multiselect
st.header("Wins vs Average PF Quadrant")
years_selected = st.multiselect(
    "Select Years", 
    options=["Combined"] + list(LEAGUE_IDS_BY_YEAR.keys()), 
    default=["Combined"]
)
if "Combined" in years_selected:
    plot_quadrant(reg_stats, playoff_stats, "Combined Wins vs Avg PF")
for year in years_selected:
    if year != "Combined":
        yr_stats = track_regular_season_only_stats({year: LEAGUE_IDS_BY_YEAR[year]})
        yr_playoffs = track_playoff_only_stats({year: LEAGUE_IDS_BY_YEAR[year]})
        plot_quadrant(yr_stats, yr_playoffs, f"🏈 {year} Wins vs Avg PF")

# Head-to-head Heatmap
st.header("Head-to-Head Regular Season Wins")
teams, mat = get_head_to_head_heatmap()

# All-time Playoff Wins Table
st.header("All-Time Playoff Wins / Appearances")
df_playoff = get_playoff_wins_table(playoff_stats)
st.table(df_playoff)


# Trades table (real data)
st.header("All Trades")
df_trades = build_detailed_ledger(LEAGUE_IDS_BY_YEAR)
if not df_trades.empty:
    st.dataframe(df_trades, use_container_width=True)
else:
    st.write("No trades found.")
