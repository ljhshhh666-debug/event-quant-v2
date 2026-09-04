
import math, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title="Event Quant V2", page_icon="📱", layout="centered")

BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://data-api.binance.vision",
]
DATA = Path("data")
DATA.mkdir(exist_ok=True)
LOG = DATA / "predictions.csv"

# ---------- API ----------
def api_get(path, params=None, timeout=8):
    err = None
    for base in BASES:
        try:
            r = requests.get(base + path, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
    raise RuntimeError(f"暂时无法连接 Binance 公共行情：{err}")

@st.cache_data(ttl=5, show_spinner=False)
def get_recent(symbol, limit=500):
    raw = api_get("/api/v3/klines", {"symbol": symbol, "interval":"1m", "limit":limit})
    return parse_klines(raw)

def parse_klines(raw):
    cols=["open_time","open","high","low","close","volume","close_time",
          "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    d=pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]:
        d[c]=pd.to_numeric(d[c], errors="coerce")
    d["open_time"]=pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return d

def download_history(symbol, bars=6000):
    # Binance REST max 1000 per request; page backwards.
    chunks=[]
    end=None
    left=bars
    while left>0:
        lim=min(1000,left)
        params={"symbol":symbol,"interval":"1m","limit":lim}
        if end is not None:
            params["endTime"]=end
        raw=api_get("/api/v3/klines", params)
        if not raw: break
        d=parse_klines(raw)
        chunks.append(d)
        first_ms=int(d.iloc[0]["open_time"].timestamp()*1000)
        end=first_ms-1
        left-=len(d)
        if len(d)<lim: break
        time.sleep(0.05)
    if not chunks:
        raise RuntimeError("没有拿到历史K线")
    out=pd.concat(chunks, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    return out.tail(bars).reset_index(drop=True)

# ---------- Features ----------
def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def feature_frame(d):
    x=d.copy()
    x["ret1"]=x["close"].pct_change(1)
    x["ret2"]=x["close"].pct_change(2)
    x["ret3"]=x["close"].pct_change(3)
    x["ret5"]=x["close"].pct_change(5)
    x["ret10"]=x["close"].pct_change(10)
    x["ret20"]=x["close"].pct_change(20)

    x["ma7"]=x["close"].rolling(7).mean()
    x["ma25"]=x["close"].rolling(25).mean()
    x["ma99"]=x["close"].rolling(99).mean()
    x["ma7_25"]=(x["ma7"]-x["ma25"])/x["close"]
    x["ma25_99"]=(x["ma25"]-x["ma99"])/x["close"]

    e12,e26=ema(x["close"],12),ema(x["close"],26)
    macd=e12-e26
    sig=ema(macd,9)
    x["macd_hist"]=(macd-sig)/x["close"]

    delta=x["close"].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=100-(100/(1+rs))

    prev=x["close"].shift(1)
    tr=pd.concat([(x["high"]-x["low"]),
                  (x["high"]-prev).abs(),
                  (x["low"]-prev).abs()],axis=1).max(axis=1)
    x["atr_pct"]=tr.rolling(14).mean()/x["close"]

    x["vol_ratio"]=x["volume"]/x["volume"].rolling(20).mean()
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
    "ret1","ret2","ret3","ret5","ret10","ret20",
    "ma7_25","ma25_99","macd_hist","rsi","atr_pct",
    "vol_ratio","taker_ratio","range_pct","body_pct","trades_z",
    "hour_sin","hour_cos","min_sin","min_cos"
]

def train_model(d):
    f=feature_frame(d).dropna(subset=FEATURES+["target"]).copy()
    if len(f)<1000:
        raise RuntimeError("历史样本太少，至少需要约1000根1分钟K线。")
    split=int(len(f)*0.75)
    train=f.iloc[:split]
    test=f.iloc[split:]
    model=HistGradientBoostingClassifier(
        learning_rate=0.055, max_iter=180, max_leaf_nodes=15,
        l2_regularization=1.5, random_state=42
    )
    model.fit(train[FEATURES],train["target"].astype(int))
    proba=model.predict_proba(test[FEATURES])[:,1]
    pred=(proba>=0.5).astype(int)
    metrics={
        "test_n":len(test),
        "acc":accuracy_score(test["target"].astype(int),pred),
        "brier":brier_score_loss(test["target"].astype(int),proba),
    }
    cal=test[["target"]].copy()
    cal["p_up"]=proba
    return model, metrics, cal, f

def calibration_table(cal):
    bins=[0,.35,.40,.45,.50,.55,.60,.65,1.0]
    labels=["<35%","35-40%","40-45%","45-50%","50-55%","55-60%","60-65%",">=65%"]
    c=cal.copy()
    c["bin"]=pd.cut(c["p_up"],bins=bins,labels=labels,include_lowest=True,right=False)
    rows=[]
    for label in labels:
        g=c[c["bin"]==label]
        if len(g):
            rows.append({
                "模型区间":label,
                "样本":len(g),
                "实际上涨率":f"{g['target'].mean()*100:.1f}%"
            })
    return pd.DataFrame(rows)

def empirical_same_grade(cal, p, direction):
    # Find test probabilities within +/- 0.04; direction-aware correctness.
    lo=max(0,p-0.04); hi=min(1,p+0.04)
    if direction=="CALL":
        g=cal[(cal.p_up>=lo)&(cal.p_up<=hi)]
        if len(g)==0: return 0, None
        return len(g), float(g.target.mean())
    else:
        # target=0 means PUT wins; compare p_down = 1-p_up
        pdn=1-cal.p_up
        g=cal[(pdn>=lo)&(pdn<=hi)]
        if len(g)==0: return 0, None
        return len(g), float((1-g.target).mean())

def breakeven(payout):
    return 100/(100+payout)

def load_log():
    if LOG.exists():
        try:return pd.read_csv(LOG)
        except:return pd.DataFrame()
    return pd.DataFrame()

def save_prediction(symbol, direction, price, p, payout):
    df=load_log()
    row=pd.DataFrame([{
        "time_utc":datetime.now(timezone.utc).isoformat(),
        "symbol":symbol,"direction":direction,"entry_price":price,
        "model_prob":p,"payout_pct":payout,"expiry_min":10
    }])
    pd.concat([df,row],ignore_index=True).to_csv(LOG,index=False)

# ---------- UI ----------
st.title("📱 Event Quant V2")
st.caption("手机友好｜BTC/ETH 10分钟方向模型｜不会自动下真钱")

with st.expander("⚙️ 设置", expanded=False):
    symbol=st.selectbox("币种",["BTCUSDT","ETHUSDT"])
    payout=st.number_input("当前净盈利 payout（%）",10.0,200.0,80.0,1.0)
    min_extra=st.slider("至少高于保本胜率多少个百分点才显示“达到条件”",0.0,15.0,5.0,0.5)
    bars=st.select_slider("训练历史长度",options=[3000,4000,5000,6000],value=5000)
    st.caption("第一次训练会下载历史1分钟K线。以后刷新实时行情不需要每次重训。")

@st.cache_resource(show_spinner=False)
def cached_train(symbol,bars):
    hist=download_history(symbol,bars)
    return train_model(hist)

if st.button("① 第一次使用：训练 / 更新模型", use_container_width=True, type="primary"):
    with st.spinner("正在下载历史行情并训练模型..."):
        cached_train.clear()
        model,metrics,cal,ff=cached_train(symbol,bars)
    st.success("模型已训练完成。")

try:
    model,metrics,cal,ff=cached_train(symbol,bars)
except Exception as e:
    st.error(str(e))
    st.stop()

recent=get_recent(symbol,300)
rf=feature_frame(recent)
latest=rf.dropna(subset=FEATURES).iloc[-1]
p_up=float(model.predict_proba(latest[FEATURES].to_frame().T)[:,1][0])
direction="CALL" if p_up>=0.5 else "PUT"
p=max(p_up,1-p_up)
be=breakeven(payout)
edge=p-be
qualified=edge >= min_extra/100

price=float(latest["close"])
n_same,real_rate=empirical_same_grade(cal,p,direction)

st.divider()
st.subheader(f"{symbol.replace('USDT','')} · 10分钟")

c1,c2=st.columns(2)
with c1:
    st.metric("当前价格",f"{price:,.2f}")
with c2:
    st.metric("方向", "▲ CALL" if direction=="CALL" else "▼ PUT")

st.markdown(f"## 模型：{p*100:.1f}%")
st.progress(int(p*100))

if qualified:
    st.success("🟢 达到你设定的观察条件")
else:
    st.error("🔴 未达到条件：不交易")

st.write(
    f"保本胜率 **{be*100:.2f}%** ｜ "
    f"模型优势 **{edge*100:+.2f} 个百分点**"
)

if real_rate is not None and n_same>=20:
    st.info(f"历史样本外测试中，与当前模型概率接近的 {n_same} 次信号，实际方向命中率约 **{real_rate*100:.1f}%**。")
else:
    st.info("历史同等级信号样本暂时不足，当前只显示模型估计，不把它当成已验证胜率。")

if st.button("记录这一次预测（模拟）",use_container_width=True):
    save_prediction(symbol,direction,price,p,payout)
    st.success("已记录。10分钟后可再回来对照价格。")

with st.expander("📊 详细数据（看不懂可以不用管）"):
    st.write(f"样本外测试数量：{metrics['test_n']}")
    st.write(f"样本外方向准确率：{metrics['acc']*100:.1f}%")
    st.write(f"Brier概率误差：{metrics['brier']:.3f}（越低越好）")
    st.dataframe(calibration_table(cal),hide_index=True,use_container_width=True)
    chart=recent.tail(120).set_index("open_time")[["close"]]
    st.line_chart(chart)
    st.caption("V2 使用时间顺序切分：前75%训练，后25%测试，避免直接拿同一段训练数据报告成绩。")

with st.expander("🧾 我的模拟预测记录"):
    lg=load_log()
    if len(lg):
        st.dataframe(lg.tail(30).iloc[::-1],hide_index=True,use_container_width=True)
    else:
        st.caption("还没有记录。")

st.divider()
st.caption("重要：模型只是辅助判断，不保证盈利；事件合约可能损失整笔下注金额。不要输入 Binance 密码、验证码、Session 或 API Key。")
