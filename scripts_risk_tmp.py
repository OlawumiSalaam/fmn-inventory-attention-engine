from pathlib import Path
import pandas as pd, numpy as np
from src.risk import RiskConfig, assess_sku

ROOT=Path('/mnt/data/fmn_project1')
df=pd.read_csv(ROOT/'data/processed/clean_daily_data.csv', parse_dates=['date'])
df=df.sort_values(['sku_id','date'])
print(df.shape, df['date'].min(), df['date'].max())

# Build a simple champion daily forecast from weekday-adjusted 28-day history.
def forecast_at(hist):
    h=hist.dropna().tail(28)
    if len(h)<14: return np.nan
    overall=h.mean()
    if overall<=0: return 0.0
    dow=h.groupby(hist.loc[h.index,'date'].dt.dayofweek).mean()
    today_dow=(hist['date'].iloc[-1]+pd.Timedelta(days=1)).dayofweek
    adj=dow.get(today_dow, overall)
    # shrink weekday effect to avoid instability
    return float(overall * (0.5 + 0.5*(adj/overall)))

origin=pd.Timestamp('2026-06-29')
rows=[]
for sku,g in df.groupby('sku_id'):
    g=g.sort_values('date')
    hist=g[g.date<=origin]
    if hist['lead_time_days'].dropna().empty: continue
    # use 28-day mean as stable daily forecast for runtime risk assessment
    fc=float(hist['units_sold'].tail(28).mean())
    if pd.isna(fc) or fc<=0: continue
    try: rows.append(assess_sku(hist,origin,fc,RiskConfig()))
    except Exception as e: print('ERR',sku,e)
out=pd.DataFrame(rows)
out.to_csv(ROOT/'artifacts/evaluation/risk_current_assessment.csv',index=False)
print(out[['sku_id','risk_state','current_stock','lead_time_days','coverage_days','expected_delivery_date','days_to_projected_stockout','projected_unmet_units']].sort_values(['risk_state','projected_unmet_units'],ascending=[True,False]).to_string(index=False))
