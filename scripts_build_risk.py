"""Build and backtest the FMN Project 1 inventory risk engine."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.forecast import build_supervised_frame, fit_model, ForecastConfig, FEATURES, predict_lead_time_demand
from src.risk import RiskConfig, assess_sku

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'data/processed/clean_daily_data.csv', parse_dates=['date'])
df=df.sort_values(['sku_id','date']).reset_index(drop=True)

# Train the champion forecast model using origins before the fixed test period.
supervised=build_supervised_frame(df)
test_start=pd.Timestamp('2026-04-30')
train=supervised[supervised['origin_date'] < test_start].copy()
model=fit_model(train, ForecastConfig())
category_map=(supervised[['category','category_code']].drop_duplicates().set_index('category')['category_code'].to_dict())

# Build one feature row at an origin using only history available at that date.
def feature_row(hist: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame | None:
    h=hist.sort_values('date').reset_index(drop=True)
    if len(h)<28: return None
    s=h['units_sold'].astype(float)
    if s.tail(28).isna().any(): return None
    lead=int(round(h.iloc[-1]['lead_time_days']))
    if lead<=0: return None
    base=s.tail(28).mean()
    if base<=0 or pd.isna(base): return None
    return pd.DataFrame([{
        'sku_id':h.iloc[-1]['sku_id'],'origin_date':origin,'category':h.iloc[-1]['category'],
        'lag_1':s.iloc[-1],'lag_7':s.iloc[-7],'lag_14':s.iloc[-14],
        'roll_mean_7':s.tail(7).mean(),'roll_mean_14':s.tail(14).mean(),'roll_mean_28':base,
        'roll_std_7':s.tail(7).std(),'roll_std_14':s.tail(14).std(),'roll_std_28':s.tail(28).std(),
        'trend_ratio_14_28':s.tail(14).mean()/base,'day_of_week':origin.dayofweek,
        'lead_time_days':lead,'category_code':category_map.get(h.iloc[-1]['category'],0),'baseline_28':base*lead
    }])

# Current as-of assessment using final model where possible.
current_date=df['date'].max()
current=[]
for sku,g in df.groupby('sku_id'):
    hist=g[g.date<=current_date].copy()
    if hist['lead_time_days'].dropna().empty or hist['closing_stock'].isna().all(): continue
    feat=feature_row(hist,current_date)
    if feat is None: continue
    ltd=float(predict_lead_time_demand(model,feat).iloc[0])
    daily=ltd/max(int(round(feat['lead_time_days'].iloc[0])),1)
    rec=assess_sku(hist,current_date,daily,RiskConfig())
    rec['forecast_method']='Pooled LightGBM'
    rec['forecast_lead_time_demand_model']=ltd
    current.append(rec)
current_df=pd.DataFrame(current)
current_df.to_csv(ROOT/'artifacts/evaluation/risk_current_assessment.csv',index=False)

# Backtest candidate risk engine. We use origins where model features are available.
origins=pd.date_range(test_start, current_date - pd.Timedelta(days=14), freq='D')
results=[]
for origin in origins:
    for sku,g in df.groupby('sku_id'):
        hist=g[g.date<=origin].copy()
        if hist.empty or hist['lead_time_days'].dropna().empty: continue
        lead=int(round(hist.iloc[-1]['lead_time_days']))
        if lead<=0 or lead not in {3,5,7,10,14}: continue
        feat=feature_row(hist,origin)
        if feat is None: continue
        pred_ltd=float(predict_lead_time_demand(model,feat).iloc[0])
        daily=pred_ltd/lead
        current_stock=float(hist.iloc[-1]['closing_stock']) if pd.notna(hist.iloc[-1]['closing_stock']) else np.nan
        if pd.isna(current_stock): continue
        # Baseline flag.
        baseline_flag=current_stock < pred_ltd
        # Candidate uses deterministic risk engine; Watch/Critical are attention states.
        rec=assess_sku(hist,origin,daily,RiskConfig(review_window_days=3,uncertainty_multiplier=0.5,overstock_multiplier=1.5,persistence_days=1))
        candidate_flag=rec['risk_state'] in {'Critical','Watch'}
        future=g[(g.date>origin)&(g.date<=origin+pd.Timedelta(days=lead))].sort_values('date').copy()
        if len(future)<lead: continue
        # Forward event: stockout occurs within lead time. Existing zero stock is already an immediate event.
        event=bool((future['closing_stock']<=0).any())
        # A true warning event is stronger if the SKU is not already at zero at origin.
        forward_warning_event=event and current_stock>0
        stockout_dates=future.loc[future['closing_stock']<=0,'date']
        warning_lead_days=float((stockout_dates.iloc[0]-origin).days) if not stockout_dates.empty else np.nan
        results.append({
            'origin_date':origin,'sku_id':sku,'lead_time_days':lead,'current_stock':current_stock,
            'predicted_lead_demand':pred_ltd,'baseline_flag':baseline_flag,'candidate_flag':candidate_flag,
            'event_next_lead':event,'forward_warning_event':forward_warning_event,
            'warning_lead_days':warning_lead_days,'risk_state':rec['risk_state']
        })

bt=pd.DataFrame(results)
bt.to_csv(ROOT/'artifacts/evaluation/risk_backtest_rows.csv',index=False)

def metrics(frame, flag):
    y=frame['event_next_lead'].astype(bool); p=frame[flag].astype(bool)
    tp=int((p&y).sum()); fp=int((p&~y).sum()); fn=int((~p&y).sum()); tn=int((~p&~y).sum())
    recall=tp/(tp+fn) if tp+fn else np.nan
    precision=tp/(tp+fp) if tp+fp else np.nan
    false_alert_rate=fp/(fp+tn) if fp+tn else np.nan
    attention_volume=float(p.mean())
    forward=frame[frame['forward_warning_event']]
    warned=forward[forward[flag]]
    warning_share=len(warned)/len(forward) if len(forward) else np.nan
    lead=float(warned['warning_lead_days'].mean()) if len(warned) else np.nan
    return dict(n=len(frame),tp=tp,fp=fp,fn=fn,tn=tn,recall=recall,precision=precision,false_alert_rate=false_alert_rate,attention_volume=attention_volume,forward_event_warning_share=warning_share,mean_actual_stockout_lead_days_when_flagged=lead)

metric_rows=[]
for lead in [3,5,7,10,14,'All']:
    f=bt if lead=='All' else bt[bt.lead_time_days==lead]
    for name,flag in [('Baseline', 'baseline_flag'),('Candidate','candidate_flag')]:
        m=metrics(f,flag); m['lead_time']=lead; m['method']=name; metric_rows.append(m)
metrics_df=pd.DataFrame(metric_rows)
metrics_df.to_csv(ROOT/'artifacts/evaluation/risk_model_comparison.csv',index=False)

# Sensitivity analysis for review window and uncertainty buffer.
sens=[]
for review in [1,3,7]:
    for um in [0.0,0.25,0.5,0.75]:
        rows=[]
        for origin in origins:
            for sku,g in df.groupby('sku_id'):
                hist=g[g.date<=origin].copy()
                if hist.empty or hist['lead_time_days'].dropna().empty: continue
                lead=int(round(hist.iloc[-1]['lead_time_days']))
                if lead not in {3,5,7,10,14}: continue
                feat=feature_row(hist,origin)
                if feat is None: continue
                pred=float(predict_lead_time_demand(model,feat).iloc[0]); daily=pred/lead
                if pd.isna(hist.iloc[-1]['closing_stock']): continue
                rec=assess_sku(hist,origin,daily,RiskConfig(review_window_days=review,uncertainty_multiplier=um,overstock_multiplier=1.5))
                future=g[(g.date>origin)&(g.date<=origin+pd.Timedelta(days=lead))]
                if len(future)<lead: continue
                rows.append({'candidate_flag':rec['risk_state'] in {'Critical','Watch'},'event':bool((future.closing_stock<=0).any())})
        sf=pd.DataFrame(rows)
        tp=((sf.candidate_flag)&(sf.event)).sum(); fp=((sf.candidate_flag)&(~sf.event)).sum(); fn=((~sf.candidate_flag)&(sf.event)).sum(); tn=((~sf.candidate_flag)&(~sf.event)).sum()
        sens.append({'review_window_days':review,'uncertainty_multiplier':um,'n':len(sf),'recall':tp/(tp+fn) if tp+fn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'false_alert_rate':fp/(fp+tn) if fp+tn else np.nan,'attention_volume':sf.candidate_flag.mean()})
pd.DataFrame(sens).to_csv(ROOT/'artifacts/evaluation/risk_parameter_sensitivity.csv',index=False)
# Driver-ready current output.
print('\nCURRENT ASSESSMENT')
print(current_df[['sku_id','risk_state','current_stock','lead_time_days','coverage_days','expected_delivery_date','days_to_projected_stockout','projected_unmet_units']].sort_values(['risk_state','projected_unmet_units'],ascending=[True,False]).to_string(index=False))
print('\nRISK COMPARISON')
print(metrics_df[metrics_df.lead_time=='All'][['method','recall','precision','false_alert_rate','attention_volume','forward_event_warning_share']].to_string(index=False))
print('\nREVIEW WINDOW')
print(pd.DataFrame(sens).to_string(index=False))
