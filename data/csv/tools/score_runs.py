import numpy as np, pandas as pd
B='experiments/PIT_WEEKLY_{t}_SUBSTITUTED_20230101_20260915_predfrom_20240102/PIT_WEEKLY_{t}_SUBSTITUTED_20230101_20260915_predfrom_20240102.parquet'
RUNS={'ALL  (340)':B.format(t='20d'),'TECH ( 91)':B.format(t='20d_TECHONLY'),'FUND (269)':B.format(t='20d_FUNDONLY')}
d={}
for t,p in RUNS.items():
    x=pd.read_parquet(p,columns=['Date','Symbol','RefPriceClose','CompanyMarketCap','predicted_return'],engine='pyarrow')
    x['Date']=pd.to_datetime(x['Date']); x=x.sort_values(['Symbol','Date'],kind='stable').reset_index(drop=True)
    g=x.groupby('Symbol',sort=False)['RefPriceClose']
    x['y']=((g.shift(-21)/g.shift(-1)-1)*100).replace([np.inf,-np.inf],np.nan)
    m=x['CompanyMarketCap'].notna(); x['mc']=np.nan
    x.loc[m,'mc']=x.loc[m].groupby('Date')['CompanyMarketCap'].rank(ascending=False,method='first')
    d[t]=x[x['predicted_return'].notna()&x['y'].notna()]
end=min(v['Date'].max() for v in d.values())
for k in d: d[k]=d[k][(d[k]['Date']>=pd.Timestamp('2024-01-02'))&(d[k]['Date']<=end)]
print(f"realised 20d returns, 2024-01-02..{end:%Y-%m-%d}\n")
def ic(w): return w.groupby('Date').apply(lambda s: s['predicted_return'].corr(s['y'],method='spearman') if len(s)>=20 else np.nan).dropna()
def bk(w,n):
    e=[]
    for _,s in w.groupby('Date'):
        if len(s)>=max(100,n*2): e.append(s.nlargest(n,'predicted_return')['y'].mean()-s['y'].mean())
    return np.array(e)
print(f"{'universe':<10}{'run':<13}{'IC':>9}{'%+d':>6}{'exc50':>9}{'exc30':>9}")
print('-'*56)
for b in [150,400,500]:
    for t in d:
        w=d[t][d[t]['mc']<=b]; i=ic(w); e50=bk(w,50); e30=bk(w,30)
        print(f"{('top-'+str(b)):<10}{t:<13}{i.mean():>+9.4f}{100*(i>0).mean():>5.0f}%"
              f"{(e50.mean() if len(e50) else np.nan):>+8.2f}%{(e30.mean() if len(e30) else np.nan):>+8.2f}%")
    print()
print("top-400 IC by year:")
for y in [2024,2025,2026]:
    line=f"  {y} "
    for t in d:
        w=d[t][(d[t]['mc']<=400)&(d[t]['Date'].dt.year==y)]
        line+=f"  {t.split()[0]}={ic(w).mean():+.4f}"
    print(line)
print("\nrank correlation between signals:")
k=list(d)
mm=d[k[0]][['Date','Symbol','predicted_return']].rename(columns={'predicted_return':k[0]})
for t in k[1:]:
    mm=mm.merge(d[t][['Date','Symbol','predicted_return']].rename(columns={'predicted_return':t}),on=['Date','Symbol'])
print(mm[k].corr(method='spearman').round(3).to_string())
