"""Build and backtest the FMN Project 1 inventory risk engine efficiently."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.forecast import build_supervised_frame, fit_model, ForecastConfig, FEATURES, predict_lead_time_demand
from src.risk import RiskConfig, assess_sku

ROOT=Path('/mnt/data/fmn_project1')
df=pd.read_csv(ROOT/'data/processed/clean_daily_data.csv', parse_dates=['date']).sort_values(['sku_id','date'])
# Final production model is fitted on all eligible historical supervised rows.
supervised=build_supervised_frame(df)
model=fit_model(supervised, ForecastConfig())
category_map=(supervised[['category','category_code']].drop_duplicates().set_index('category')['category_code'].to_dict())

def feature_row(hist, origin):
    h=hist.sort_values('date').reset_index(drop=True)
    if len(h)<28: return None
    s=h.units_sold.astype(float)
    if s.tail(28).isna().any(): return None
    lead=int(round(h.iloc[-1].lead_time_days))
    if lead not in {3,5,7,10,14}: return None
    base=s.tail(28).mean()
    if not np.isfinite(base) or base<=0: return None
    return {
        'sku_id':h.iloc[-1].sku_id,'origin_date':origin,'category':h.iloc[-1].category,
        'lag_1':s.iloc[-1],'lag_7':s.iloc[-7],'lag_14':s.iloc[-14],
        'roll_mean_7':s.tail(7).mean(),'roll_mean_14':s.tail(14).mean(),'roll_mean_28':base,
        'roll_std_7':s.tail(7).std(),'roll_std_14':s.tail(14).std(),'roll_std_28':s.tail(28).std(),
        'trend_ratio_14_28':s.tail(14).mean()/base,'day_of_week':origin.dayofweek,
        'lead_time_days':lead,'category_code':category_map.get(h.iloc[-1].category,0),'baseline_28':base*lead,
        'current_stock':h.iloc[-1].closing_stock,
    }

# Current assessment as of dataset end.
current_date=df.date.max(); current=[]
for sku,g in df.groupby('sku_id'):
    hist=g[g.date<=current_date]
    fr=feature_row(hist,current_date)
    if fr is None or pd.isna(fr['current_stock']): continue
    feat=pd.DataFrame([fr])
    ltd=float(predict_lead_time_demand(model,feat).iloc[0]); daily=ltd/fr['lead_time_days']
    rec=assess_sku(hist,current_date,daily,RiskConfig())
    rec['forecast_method']='Pooled LightGBM'; rec['forecast_lead_time_demand_model']=ltd
    current.append(rec)
current_df=pd.DataFrame(current)
current_df.to_csv(ROOT/'artifacts/evaluation/risk_current_assessment.csv',index=False)

# Prepare all backtest feature rows and batch predict.
test_start=pd.Timestamp('2026-04-30'); origins=pd.date_range(test_start,current_date-pd.Timedelta(days=14),freq='D')
rows=[]
for origin in origins:
    for sku,g in df.groupby('sku_id'):
        hist=g[g.date<=origin]
        fr=feature_row(hist,origin)
        if fr is None or pd.isna(fr['current_stock']): continue
        future=g[(g.date>origin)&(g.date<=origin+pd.Timedelta(days=int(fr['lead_time_days'])))].sort_values('date')
        if len(future)<fr['lead_time_days']: continue
        fr['event_next_lead']=bool((future.closing_stock<=0).any())
        fr['forward_warning_event']=fr['event_next_lead'] and fr['current_stock']>0
        stockouts=future.loc[future.closing_stock<=0,'date']
        fr['warning_lead_days']=float((stockouts.iloc[0]-origin).days) if not stockouts.empty else np.nan
        rows.append(fr)
bt=pd.DataFrame(rows)
preds=predict_lead_time_demand(model,bt)
bt['predicted_lead_demand']=preds.to_numpy()
bt['baseline_flag']=bt.current_stock < bt.predicted_lead_demand

# Precompute expected replenishment and uncertainty. Candidate risk uses stock trajectory up to lead time + review window.
next_dates=[]; receipt_qtys=[]; uncertainties=[]
for (sku,origin),grp in df.groupby(['sku_id','date']): pass
for _,r in bt.iterrows():
    hist=df[(df.sku_id==r.sku_id)&(df.date<=r.origin_date)]
    recs=hist[hist.units_received>0][['date','units_received']].sort_values('date')
    dates=recs.date.drop_duplicates().sort_values(); gaps=dates.diff().dt.days.dropna()
    cadence=float(gaps.median()) if not gaps.empty else np.nan
    qty=float(recs.tail(5).units_received.median()) if not recs.empty else 0.0
    if np.isfinite(cadence) and cadence>0 and not dates.empty:
        nd=dates.iloc[-1]+pd.Timedelta(days=int(round(cadence)))
        while nd<=r.origin_date: nd += pd.Timedelta(days=int(round(cadence)))
    else: nd=pd.NaT
    s=hist.units_sold.dropna().astype(float).tail(28)
    unc=float(s.std(ddof=1)) if len(s)>=7 else 0.0
    next_dates.append(nd); receipt_qtys.append(qty); uncertainties.append(unc)
bt['expected_delivery_date']=next_dates; bt['expected_receipt_qty']=receipt_qtys; bt['uncertainty_daily']=uncertainties

# Vectorised risk classifier.
def add_candidate(frame, review, um):
    out=frame.copy(); states=[]
    for r in out.itertuples(index=False):
        lead=int(r.lead_time_days); daily=r.predicted_lead_demand/lead; stock=float(r.current_stock)
        safety=um*r.uncertainty_daily*np.sqrt(max(lead,1))
        nd=r.expected_delivery_date; qty=float(r.expected_receipt_qty or 0)
        # Simulate only until lead + review horizon.
        horizon=max(lead+review,7); projected=[]; st=stock
        for d in range(1,horizon+1):
            date=r.origin_date+pd.Timedelta(days=d)
            receipt=qty if pd.notna(nd) and date==nd else 0.0
            st=st+receipt-(daily+um*r.uncertainty_daily)
            projected.append((date,st,receipt))
        if stock<=0: state='Critical'
        else:
            pre=[x for x in projected if x[2]==0]
            state='Critical' if any(x[1]<=0 for x in pre) else ('Watch' if any(x[1]<=safety for x in projected[:review]) else ('Overstock' if projected[-1][1]>1.5*(daily*lead)+safety else 'Healthy'))
        states.append(state)
    out[f'candidate_{review}_{um}']=pd.Series(states,index=out.index).isin(['Critical','Watch'])
    return out

for review in [1,3,7]:
    for um in [0.0,0.25,0.5,0.75]: bt=add_candidate(bt,review,um)

def metrics(frame,flag):
    y=frame.event_next_lead.astype(bool); p=frame[flag].astype(bool)
    tp=int((p&y).sum()); fp=int((p&~y).sum()); fn=int((~p&y).sum()); tn=int((~p&~y).sum())
    forward=frame[frame.forward_warning_event]
    warned=forward[forward[flag]]
    return {'n':len(frame),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'recall':tp/(tp+fn) if tp+fn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'false_alert_rate':fp/(fp+tn) if fp+tn else np.nan,'attention_volume':p.mean(),'forward_event_warning_share':len(warned)/len(forward) if len(forward) else np.nan,'mean_actual_stockout_lead_days_when_flagged':warned.warning_lead_days.mean() if len(warned) else np.nan}

metric_rows=[]
for lead in [3,5,7,10,14,'All']:
    f=bt if lead=='All' else bt[bt.lead_time_days==lead]
    if lead=='All':
        for name,flag in [('Baseline','baseline_flag'),('Candidate','candidate_3_0.5')]:
            m=metrics(f,flag); m.update(lead_time=lead,method=name); metric_rows.append(m)
    else:
        for name,flag in [('Baseline','baseline_flag'),('Candidate','candidate_3_0.5')]:
            m=metrics(f,flag); m.update(lead_time=lead,method=name); metric_rows.append(m)
metrics_df=pd.DataFrame(metric_rows); metrics_df.to_csv(ROOT/'artifacts/evaluation/risk_model_comparison.csv',index=False)

sens=[]
for review in [1,3,7]:
    for um in [0.0,0.25,0.5,0.75]:
        m=metrics(bt,f'candidate_{review}_{um}'); m.update(review_window_days=review,uncertainty_multiplier=um); sens.append(m)
sens_df=pd.DataFrame(sens); sens_df.to_csv(ROOT/'artifacts/evaluation/risk_parameter_sensitivity.csv',index=False)
bt.to_csv(ROOT/'artifacts/evaluation/risk_backtest_rows.csv',index=False)

print('CURRENT')
print(current_df[['sku_id','risk_state','current_stock','lead_time_days','coverage_days','expected_delivery_date','days_to_projected_stockout','projected_unmet_units']].sort_values(['risk_state','projected_unmet_units'],ascending=[True,False]).to_string(index=False))
print('\nCOMPARISON')
print(metrics_df[metrics_df.lead_time=='All'][['method','recall','precision','false_alert_rate','attention_volume','forward_event_warning_share']].to_string(index=False))
print('\nSENSITIVITY')
print(sens_df[['review_window_days','uncertainty_multiplier','recall','precision','false_alert_rate','attention_volume']].to_string(index=False))
