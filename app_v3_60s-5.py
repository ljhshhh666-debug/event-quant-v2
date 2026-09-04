
import math, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title="Event Quant V3", page_icon="📱", layout="centered")

# 每 60 秒自动刷新整个页面：重新获取最新行情并重新计算预测，
# 但 cached_train 使用 st.cache_resource，因此不会每分钟重新训练模型。
components.html(
    """
    <script>
    setTimeout(function () {
        window.parent.location.reload();
    }, 60000);
    </script>
    """,
    height=0,
)

BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://data-api.binance.vision",
]
DATA = Path("data")
DATA.mkdir(exist_ok=True)
LOG = DATA / "predictions_v3.csv"

def api_get(path, params=None, timeout=8):
    err=None
    for base in BASES:
        try:
            r=requests.get(base+path,params=params,timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err=e
    raise RuntimeError(f"暂时无法连接 Binance 公共行情：{err}")

def parse_klines(raw):
    cols=["open_time","open","high","low","close","volume","close_time",
          "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    d=pd.DataFrame(raw,columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["open_time"]=pd.to_datetime(d["open_time"],unit="ms",utc=True)
    return d

@st.cache_data(ttl=5,show_spinner=False)
def get_recent(symbol,limit=400):
    return parse_klines(api_get("/api/v3/klines",{"symbol":symbol,"interval":"1m","limit":limit}))

def download_history(symbol,bars=12000):
    chunks=[]
    end=None
    left=bars
    while left>0:
        lim=min(1000,left)
        p={"symbol":symbol,"interval":"1m","limit":lim}
        if end is not None: p["endTime"]=end
        raw=api_get("/api/v3/klines",p)
        if not raw: break
        d=parse_klines(raw)
        chunks.append(d)
        end=int(d.iloc[0]["open_time"].timestamp()*1000)-1
        left-=len(d)
        if len(d)<lim: break
        time.sleep(0.04)
    out=pd.concat(chunks,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    return out.tail(bars).reset_index(drop=True)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def feature_frame(d):
    x=d.copy()
    for n in [1,2,3,5,10,20,30]:
        x[f"ret{n}"]=x["close"].pct_change(n)
    x["ma7"]=x["close"].rolling(7).mean()
    x["ma25"]=x["close"].rolling(25).mean()
    x["ma99"]=x["close"].rolling(99).mean()
    x["ma7_25"]=(x["ma7"]-x["ma25"])/x["close"]
    x["ma25_99"]=(x["ma25"]-x["ma99"])/x["close"]
    e12,e26=ema(x["close"],12),ema(x["close"],26)
    macd=e12-e26
    x["macd_hist"]=(macd-ema(macd,9))/x["close"]

    delta=x["close"].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=100-(100/(1+rs))

    prev=x["close"].shift(1)
    tr=pd.concat([
        x["high"]-x["low"],
        (x["high"]-prev).abs(),
        (x["low"]-prev).abs()
    ],axis=1).max(axis=1)
    x["atr_pct"]=tr.rolling(14).mean()/x["close"]
    x["vol_ratio"]=x["volume"]/x["volume"].rolling(20).mean()
    x["vol_ratio60"]=x["volume"]/x["volume"].rolling(60).mean()
    x["taker_ratio"]=x["taker_buy_base"]/x["volume"].replace(0,np.nan)
    x["range_pct"]=(x["high"]-x["low"])/x["close"]
    x["body_pct"]=(x["close"]-x["open"])/x["open"]
    x["trades_z"]=(x["trades"]-x["trades"].rolling(30).mean())/x["trades"].rolling(30).std()

    x["hour"]=x["open_time"].dt.hour
    x["minute"]=x["open_time"].dt.minute
    x["hour_sin"]=np.sin(2*np.pi*x["hour"]/24)
    x["hour_cos"]=np.cos(2*np.pi*x["hour"]/24)
    x["min_sin"]=np.sin(2*np.pi*x["minute"]/60)
    x["min_cos"]=np.cos(2*np.pi*x["minute"]/60)

    x["future_close"]=x["close"].shift(-10)
    x["target"]=(x["future_close"]>x["close"]).astype(float)
    x.loc[x["future_close"].isna(),"target"]=np.nan
    return x

FEATURES=[
    "ret1","ret2","ret3","ret5","ret10","ret20","ret30",
    "ma7_25","ma25_99","macd_hist","rsi","atr_pct",
    "vol_ratio","vol_ratio60","taker_ratio","range_pct","body_pct","trades_z",
    "hour_sin","hour_cos","min_sin","min_cos"
]

def fit_v3(d):
    f=feature_frame(d).dropna(subset=FEATURES+["target"]).copy()
    n=len(f)
    if n<4000:
        raise RuntimeError("有效历史样本不足。")
    i1=int(n*0.60)
    i2=int(n*0.80)
    train=f.iloc[:i1]
    val=f.iloc[i1:i2]
    test=f.iloc[i2:]

    base=HistGradientBoostingClassifier(
        learning_rate=0.045,max_iter=260,max_leaf_nodes=15,
        l2_regularization=2.0,random_state=42
    )
    base.fit(train[FEATURES],train["target"].astype(int))

    p_val=base.predict_proba(val[FEATURES])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val,val["target"].astype(int))

    p_test_raw=base.predict_proba(test[FEATURES])[:,1]
    p_test=iso.predict(p_test_raw)
    pred=(p_test>=0.5).astype(int)
    metrics={
        "train_n":len(train),"val_n":len(val),"test_n":len(test),
        "acc":accuracy_score(test["target"].astype(int),pred),
        "brier":brier_score_loss(test["target"].astype(int),p_test),
        "raw_brier":brier_score_loss(test["target"].astype(int),p_test_raw)
    }
    scored=test[["target"]].copy()
    scored["p_up"]=p_test
    return base,iso,metrics,scored

def wilson_lower(successes,n,z=1.645):
    if n<=0: return 0.0
    phat=successes/n
    den=1+z*z/n
    center=phat+z*z/(2*n)
    adj=z*math.sqrt((phat*(1-phat)+z*z/(4*n))/n)
    return max(0.0,(center-adj)/den)

def empirical_grade(scored,p,direction,width=0.04):
    if direction=="CALL":
        q=scored[(scored["p_up"]>=max(0,p-width))&(scored["p_up"]<=min(1,p+width))]
        wins=q["target"].sum()
    else:
        pdn=1-scored["p_up"]
        q=scored[(pdn>=max(0,p-width))&(pdn<=min(1,p+width))]
        wins=(1-q["target"]).sum()
    n=len(q)
    if n==0: return 0,None,None
    rate=float(wins/n)
    low=wilson_lower(float(wins),n)
    return n,rate,low

def breakeven(payout): return 100/(100+payout)

def save_prediction(symbol,direction,price,p,emp,n):
    df=pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    row=pd.DataFrame([{
        "time_utc":datetime.now(timezone.utc).isoformat(),
        "symbol":symbol,"direction":direction,"entry_price":price,
        "calibrated_prob":p,"empirical_rate":emp,"similar_n":n
    }])
    pd.concat([df,row],ignore_index=True).to_csv(LOG,index=False)

st.title("📱 Event Quant V3")
st.caption("概率校准 + 样本外过滤｜只做辅助判断｜不会自动下真钱")
st.caption("🔄 页面每 60 秒自动刷新行情与预测｜不会自动重新训练模型")

with st.expander("⚙️ 设置"):
    symbol=st.selectbox("币种",["BTCUSDT","ETHUSDT"])
    payout=st.number_input("当前净盈利 payout（%）",10.0,200.0,80.0,1.0)
    min_extra=st.slider("最低优势（百分点）",0.0,15.0,5.0,0.5)
    min_samples=st.slider("相似历史信号最少样本",20,200,60,10)
    bars=st.select_slider("训练历史长度",options=[8000,10000,12000,15000],value=12000)
    st.caption("建议保持默认。样本越多，第一次训练时间越长。")

@st.cache_resource(show_spinner=False)
def cached_train(symbol,bars):
    hist=download_history(symbol,bars)
    return fit_v3(hist)

if st.button("① 训练 / 更新 V3 模型",use_container_width=True,type="primary"):
    with st.spinner("正在下载历史行情、训练并做概率校准..."):
        cached_train.clear()
        base,iso,metrics,scored=cached_train(symbol,bars)
    st.success("V3 模型训练完成。")

try:
    base,iso,metrics,scored=cached_train(symbol,bars)
except Exception as e:
    st.error(str(e)); st.stop()

recent=get_recent(symbol,400)
rf=feature_frame(recent).dropna(subset=FEATURES)
latest=rf.iloc[-1]
raw=float(base.predict_proba(latest[FEATURES].to_frame().T)[:,1][0])
p_up=float(iso.predict([raw])[0])
direction="CALL" if p_up>=0.5 else "PUT"
p=max(p_up,1-p_up)
price=float(latest["close"])
last_update=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
be=breakeven(payout)
edge=p-be
n_same,emp,low=empirical_grade(scored,p,direction)

model_ok=edge>=min_extra/100
emp_ok=(emp is not None and emp>=be+min_extra/100)
low_ok=(low is not None and low>=be)
samples_ok=n_same>=min_samples
qualified=model_ok and emp_ok and low_ok and samples_ok

st.divider()
st.subheader(f"{symbol.replace('USDT','')} · 10分钟")
c1,c2=st.columns(2)
c1.metric("当前价格",f"{price:,.2f}")
c2.metric("方向","▲ CALL" if direction=="CALL" else "▼ PUT")
st.markdown(f"## 校准后概率：{p*100:.1f}%")
st.caption(f"最近更新时间：{last_update}｜约 60 秒后自动刷新")
st.progress(int(p*100))

if qualified:
    st.success("🟢 达到 V3 条件：可以观察")
else:
    st.error("🔴 未达到 V3 条件：不交易")

st.write(f"保本胜率 **{be*100:.2f}%** ｜ 校准模型优势 **{edge*100:+.2f} 个百分点**")

if emp is None:
    st.info("相似历史信号不足。")
else:
    st.info(
        f"样本外相似信号 **{n_same} 次** ｜ 实际命中率 **{emp*100:.1f}%** ｜ "
        f"90%置信下限 **{low*100:.1f}%**"
    )

checks=pd.DataFrame([
    ["模型概率过线","✅" if model_ok else "❌"],
    ["相似历史命中率过线","✅" if emp_ok else "❌"],
    ["历史样本数足够","✅" if samples_ok else "❌"],
    ["置信下限不低于保本线","✅" if low_ok else "❌"],
],columns=["检查项","结果"])
st.dataframe(checks,hide_index=True,use_container_width=True)

if st.button("记录这一次预测（模拟）",use_container_width=True):
    save_prediction(symbol,direction,price,p,emp if emp is not None else np.nan,n_same)
    st.success("已记录。")

with st.expander("📊 详细测试数据"):
    st.write(f"训练样本：{metrics['train_n']}｜校准样本：{metrics['val_n']}｜最终测试样本：{metrics['test_n']}")
    st.write(f"样本外方向准确率：{metrics['acc']*100:.1f}%")
    st.write(f"校准前 Brier：{metrics['raw_brier']:.3f}")
    st.write(f"校准后 Brier：{metrics['brier']:.3f}")
    st.caption("Brier 越低越好。V3 使用独立验证段做概率校准，再用最后20%数据做最终测试。")

st.divider()
st.caption("V3 仍是实验性辅助工具。只有绿色条件出现也不代表保证盈利；建议先长期模拟记录，确认稳定后再考虑真实资金。")
