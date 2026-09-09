import io
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tradingpatterns.tradingpatterns import detect_head_shoulder, detect_double_top_bottom, detect_channel

st.set_page_config(page_title='Chart Pattern Scanner', page_icon='📈', layout='wide')
UNIVERSES={'NIFTY 50':'https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv','NIFTY Next 50':'https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv','NIFTY Midcap 150':'https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv','NIFTY Smallcap 250':'https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv'}
CANDLE_OPTIONS={'Daily':[50,100,150,200,250,300],'Weekly':[26,52,78,104,156],'Monthly':[12,24,36,60,120]}
BIAS={'Double Bottom':'Bullish','Double Top':'Bearish','Head and Shoulder':'Bearish','Inverse Head and Shoulder':'Bullish','Channel Up':'Bullish','Channel Down':'Bearish'}

def period_for(tf,n):
    return f'{max(2,int(n*1.7))}d' if tf=='Daily' else (f'{max(2,int(n*1.7/52)+1)}y' if tf=='Weekly' else f'{max(2,int(n*1.7/12)+1)}y')

@st.cache_data(ttl=86400)
def symbols(name):
    r=requests.get(UNIVERSES[name],headers={'User-Agent':'Mozilla/5.0'},timeout=20); r.raise_for_status()
    x=pd.read_csv(io.BytesIO(r.content)); c=next(c for c in x.columns if c.upper()=='SYMBOL')
    return x[c].dropna().astype(str).str.strip().tolist()

@st.cache_data(ttl=900)
def prices(symbol,tf,n):
    import yfinance as yf
    x=yf.download(symbol,period=period_for(tf,n),interval={'Daily':'1d','Weekly':'1wk','Monthly':'1mo'}[tf],auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    need={'Open','High','Low','Close'}
    return pd.DataFrame() if x.empty or not need.issubset(x.columns) else x.dropna(subset=list(need)).tail(n)

def patternpy(df):
    x=df[[c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]].copy()
    x=detect_head_shoulder(x,window=3)
    x=detect_double_top_bottom(x,window=3,threshold=0.05)
    x=detect_channel(x,window=3)
    return x

def results(x):
    out=[]
    for col in ['head_shoulder_pattern','double_pattern','channel_pattern']:
        if col in x.columns:
            for v in x[col].dropna().astype(str):
                if v in BIAS and v not in out: out.append(v)
    return out

def chart(symbol,tf,df):
    x=patternpy(df); fig=go.Figure(go.Candlestick(x=x.index,open=x.Open,high=x.High,low=x.Low,close=x.Close,name=symbol))
    for col in ['head_shoulder_pattern','double_pattern','channel_pattern']:
        if col not in x.columns: continue
        for pat in x[col].dropna().astype(str).unique():
            if pat not in BIAS: continue
            idx=x.index[x[col].astype(str)==pat]
            if len(idx)==0: continue
            vals=[float(x.loc[i,'Low'] if BIAS[pat]=='Bullish' else x.loc[i,'High']) for i in idx]
            fig.add_trace(go.Scatter(x=idx,y=vals,mode='markers+text',text=[pat]*len(idx),textposition='bottom center' if BIAS[pat]=='Bullish' else 'top center',name=pat))
    fig.update_layout(title=f'{symbol} — {tf}',xaxis_rangeslider_visible=False,height=700,margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig,use_container_width=True)

st.title('📈 Chart Pattern Scanner')
st.caption('Pattern detection powered directly by PatternPy')
with st.sidebar:
    universe=st.selectbox('Universe',list(UNIVERSES)); tf=st.selectbox('Pattern timeframe',list(CANDLE_OPTIONS))
    n=st.selectbox('Number of candles',CANDLE_OPTIONS[tf],index=CANDLE_OPTIONS[tf].index({'Daily':100,'Weekly':52,'Monthly':36}[tf]))
    direction=st.radio('Direction filter',['All','Bullish','Bearish'])
    scan=st.button('🔎 SCAN PATTERNS',type='primary',use_container_width=True)
    st.caption(f'PatternPy receives the latest {n} {tf.lower()} candles.')

if scan:
    try: syms=symbols(universe)
    except Exception as e: st.error(f'Unable to load {universe}: {e}'); st.stop()
    rows=[]; bar=st.progress(0)
    for i,s in enumerate(syms,1):
        try:
            d=prices(s+'.NS',tf,n)
            if d.empty: continue
            ps=[p for p in results(patternpy(d)) if direction=='All' or BIAS[p]==direction]
            if ps: rows.append({'Stock':s,'Pattern':', '.join(ps)})
        except Exception: pass
        bar.progress(i/len(syms))
    st.session_state.results=rows; st.session_state.selected=None

if 'results' in st.session_state:
    rows=st.session_state.results; st.subheader(f'Patterns detected ({len(rows)} stocks)')
    if not rows: st.info('No PatternPy patterns were detected for this scan.')
    else:
        h=st.columns([2,7,1]); h[0].markdown('**Stock**'); h[1].markdown('**Pattern**'); h[2].markdown('**View**')
        for r in rows:
            c=st.columns([2,7,1]); c[0].write(r['Stock']); c[1].write(r['Pattern'])
            if c[2].button('View',key='view_'+r['Stock']): st.session_state.selected=r['Stock']

if st.session_state.get('selected'):
    s=st.session_state.selected; st.divider(); st.subheader(f'{s} — PatternPy chart')
    try:
        d=prices(s+'.NS',tf,n); chart(s,tf,d); st.write('**PatternPy result:** '+(', '.join(results(patternpy(d))) or 'None'))
    except Exception as e: st.error(f'Unable to render chart: {e}')
