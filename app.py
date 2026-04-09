import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


APP_TITLE = "Semi-Auto Trading OS Mobile"
DB_PATH = Path("trade_journal.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            structural_risk_per_share REAL,
            effective_risk_per_share REAL,
            reward_per_share REAL,
            rr_ratio REAL,
            quantity INTEGER,
            calm_loss_budget REAL,
            atr REAL,
            atr_multiple REAL,
            slippage_factor REAL,
            setup_grade TEXT,
            gate_passed INTEGER,
            news_block INTEGER,
            volatility_block INTEGER,
            revenge_block INTEGER,
            sleep_block INTEGER,
            correlation_block INTEGER,
            htf_aligned INTEGER,
            role_reversal_ok INTEGER,
            volume_quality_ok INTEGER,
            room_to_run_ok INTEGER,
            time_stop_bars INTEGER,
            time_stop_triggered INTEGER,
            rule_adherence_score INTEGER,
            would_repeat_100x TEXT,
            notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


@dataclass
class TradeRecord:
    created_at: str
    symbol: str
    side: str
    entry_price: float
    stop_price: float
    target_price: float
    structural_risk_per_share: float
    effective_risk_per_share: float
    reward_per_share: float
    rr_ratio: float
    quantity: int
    calm_loss_budget: float
    atr: float
    atr_multiple: float
    slippage_factor: float
    setup_grade: str
    gate_passed: int
    news_block: int
    volatility_block: int
    revenge_block: int
    sleep_block: int
    correlation_block: int
    htf_aligned: int
    role_reversal_ok: int
    volume_quality_ok: int
    room_to_run_ok: int
    time_stop_bars: int
    time_stop_triggered: int
    rule_adherence_score: int
    would_repeat_100x: str
    notes: str


def save_trade(record: TradeRecord) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trade_journal (
            created_at, symbol, side, entry_price, stop_price, target_price,
            structural_risk_per_share, effective_risk_per_share, reward_per_share,
            rr_ratio, quantity, calm_loss_budget, atr, atr_multiple,
            slippage_factor, setup_grade, gate_passed, news_block,
            volatility_block, revenge_block, sleep_block, correlation_block,
            htf_aligned, role_reversal_ok, volume_quality_ok, room_to_run_ok,
            time_stop_bars, time_stop_triggered, rule_adherence_score,
            would_repeat_100x, notes
        ) VALUES (
            :created_at, :symbol, :side, :entry_price, :stop_price, :target_price,
            :structural_risk_per_share, :effective_risk_per_share, :reward_per_share,
            :rr_ratio, :quantity, :calm_loss_budget, :atr, :atr_multiple,
            :slippage_factor, :setup_grade, :gate_passed, :news_block,
            :volatility_block, :revenge_block, :sleep_block, :correlation_block,
            :htf_aligned, :role_reversal_ok, :volume_quality_ok, :room_to_run_ok,
            :time_stop_bars, :time_stop_triggered, :rule_adherence_score,
            :would_repeat_100x, :notes
        )
        """,
        asdict(record),
    )
    conn.commit()
    conn.close()


@st.cache_data
def load_journal() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trade_journal ORDER BY id DESC", conn)
    conn.close()
    return df


@st.cache_data
def generate_sample_ohlcv(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=rows, freq="D")

    close = [100.0]
    for _ in range(rows - 1):
        close.append(close[-1] * (1 + rng.normal(0.0008, 0.02)))
    close = np.array(close)

    open_ = np.insert(close[:-1], 0, close[0] * (1 + rng.normal(0, 0.01)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, size=rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, size=rows))
    volume = rng.integers(50000, 250000, size=rows)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


@st.cache_data
def parse_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    lower_map = {col.lower(): col for col in df.columns}

    required_variants = {
        "date": ["date", "datetime", "time", "timestamp"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close", "adj_close"],
        "volume": ["volume", "vol", "v"],
    }

    resolved = {}
    for canonical, variants in required_variants.items():
        found = None
        for variant in variants:
            if variant in lower_map:
                found = lower_map[variant]
                break
        if found is None and canonical != "volume":
            raise ValueError(f"必須列が見つかりません: {canonical}")
        resolved[canonical] = found

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[resolved["date"]]),
            "Open": pd.to_numeric(df[resolved["open"]], errors="coerce"),
            "High": pd.to_numeric(df[resolved["high"]], errors="coerce"),
            "Low": pd.to_numeric(df[resolved["low"]], errors="coerce"),
            "Close": pd.to_numeric(df[resolved["close"]], errors="coerce"),
            "Volume": pd.to_numeric(df[resolved["volume"]], errors="coerce") if resolved["volume"] else np.nan,
        }
    ).dropna(subset=["Date", "Open", "High", "Low", "Close"])

    return out.sort_values("Date").reset_index(drop=True)


@st.cache_data
def add_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(atr_period).mean()

    if df["Volume"].notna().sum() > 0:
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["CumTPV"] = (typical_price * df["Volume"].fillna(0)).cumsum()
        df["CumVol"] = df["Volume"].fillna(0).cumsum().replace(0, np.nan)
        df["VWAP"] = df["CumTPV"] / df["CumVol"]
        df["VolMA20"] = df["Volume"].rolling(20).mean()
    else:
        df["VWAP"] = np.nan
        df["VolMA20"] = np.nan

    return df


@st.cache_data
def compute_volume_profile(df: pd.DataFrame, bins: int = 24) -> tuple[pd.DataFrame, float | None]:
    if "Volume" not in df.columns or df["Volume"].notna().sum() == 0:
        return pd.DataFrame(), None

    working = df.dropna(subset=["Close", "Volume"]).copy()
    if working.empty:
        return pd.DataFrame(), None

    price_min = float(working["Low"].min())
    price_max = float(working["High"].max())
    if price_max <= price_min:
        return pd.DataFrame(), None

    edges = np.linspace(price_min, price_max, bins + 1)
    bin_idx = pd.cut(working["Close"], bins=edges, include_lowest=True, labels=False)
    profile = (
        working.assign(_bin=bin_idx)
        .dropna(subset=["_bin"])
        .groupby("_bin", as_index=False)
        .agg(volume=("Volume", "sum"))
    )

    if profile.empty:
        return pd.DataFrame(), None

    profile["_bin"] = profile["_bin"].astype(int)
    profile["price_low"] = profile["_bin"].map(lambda i: edges[i])
    profile["price_high"] = profile["_bin"].map(lambda i: edges[i + 1])
    profile["price_mid"] = (profile["price_low"] + profile["price_high"]) / 2

    poc_row = profile.loc[profile["volume"].idxmax()]
    poc_price = float(poc_row["price_mid"])
    return profile, poc_price


def infer_side(entry: float, stop: float) -> str:
    if stop < entry:
        return "Long"
    if stop > entry:
        return "Short"
    return "Flat"


def calc_risk(entry: float, stop: float, target: float, calm_loss_budget: float, slippage_factor: float) -> dict:
    structural_risk = abs(entry - stop)
    reward = abs(target - entry)
    effective_risk = structural_risk * slippage_factor
    quantity = 0 if effective_risk <= 0 else math.floor(calm_loss_budget / effective_risk)
    actual_risk_amount = quantity * effective_risk
    actual_reward_amount = quantity * reward
    rr = 0.0 if effective_risk <= 0 else reward / effective_risk

    return {
        "structural_risk_per_share": structural_risk,
        "effective_risk_per_share": effective_risk,
        "reward_per_share": reward,
        "quantity": quantity,
        "actual_risk_amount": actual_risk_amount,
        "actual_reward_amount": actual_reward_amount,
        "rr_ratio": rr,
    }


def gate_decision(news_block: bool, volatility_block: bool, revenge_block: bool, sleep_block: bool, correlation_block: bool) -> tuple[bool, list[str]]:
    reasons = []
    if news_block:
        reasons.append("重要イベント前・直後")
    if volatility_block:
        reasons.append("異常ボラティリティ")
    if revenge_block:
        reasons.append("取り返しモード")
    if sleep_block:
        reasons.append("睡眠・体調不足")
    if correlation_block:
        reasons.append("相関資産の急変")
    return len(reasons) == 0, reasons


def setup_score(htf_aligned: bool, role_reversal_ok: bool, volume_quality_ok: bool, room_to_run_ok: bool, setup_grade: str) -> int:
    score = 0
    score += 25 if htf_aligned else 0
    score += 25 if role_reversal_ok else 0
    score += 25 if volume_quality_ok else 0
    score += 25 if room_to_run_ok else 0

    if setup_grade == "A":
        score += 10
    elif setup_grade == "C":
        score -= 10

    return max(0, min(score, 100))


def calc_expected_value(rr_ratio: float, win_rate: float) -> float:
    loss_rate = 1 - win_rate
    return (win_rate * rr_ratio) - loss_rate


def build_chart(
    df: pd.DataFrame,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    time_stop_bars: int = 0,
    volume_profile_bins: int = 24,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.82, 0.18],
        horizontal_spacing=0.03,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    if "EMA20" in df.columns:
        fig.add_scatter(x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20", row=1, col=1)
    if "SMA50" in df.columns:
        fig.add_scatter(x=df["Date"], y=df["SMA50"], mode="lines", name="SMA50", row=1, col=1)
    if "VWAP" in df.columns and df["VWAP"].notna().sum() > 0:
        fig.add_scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP", row=1, col=1)

    profile_df, poc_price = compute_volume_profile(df, bins=volume_profile_bins)
    if not profile_df.empty:
        fig.add_trace(
            go.Bar(
                x=profile_df["volume"],
                y=profile_df["price_mid"],
                orientation="h",
                name="Vol Profile",
                opacity=0.4,
            ),
            row=1,
            col=2,
        )
        if poc_price is not None:
            fig.add_hline(
                y=poc_price,
                line_dash="dash",
                annotation_text="POC",
                annotation_position="top right",
            )

    for label, price in [("Entry", entry_price), ("Stop", stop_price), ("Target", target_price)]:
        if price is not None and price > 0:
            fig.add_hline(y=price, line_dash="dot", annotation_text=label, annotation_position="top left")

    if time_stop_bars > 0 and len(df) >= 2:
        diffs = df["Date"].diff().dropna()
        step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
        target_date = df["Date"].iloc[-1] + (step * time_stop_bars)
        fig.add_vline(
            x=target_date,
            line_color="orange",
            line_dash="dash",
            annotation_text="Time Stop",
            annotation_position="top",
            row=1,
            col=1,
        )

    fig.update_layout(
        title="Chart + Time Stop + Volume Profile",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=620,
        margin=dict(l=8, r=8, t=44, b=8),
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Volume by Price", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    return fig


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    init_db()

    st.title(APP_TITLE)
    st.caption("スマホのブラウザで使いやすい縦スクロール前提の半自動トレード監査ツール")

    st.markdown(
        """
        <style>
        .stButton > button {
            width: 100%;
            min-height: 3rem;
        }
        div[data-testid="stMetric"] {
            padding: 0.4rem 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("使い方", expanded=False):
        st.markdown(
            """
            1. CSVをアップロードするか、デモデータで試します。  
            2. 基本設定と売買プランを入れます。  
            3. チャートを確認し、ゲートと監査で通過可否を見ます。  
            4. 記録を保存して、規律ダッシュボードで振り返ります。  

            想定CSV列: Date, Open, High, Low, Close, Volume
            """
        )

    with st.expander("データ読み込み", expanded=True):
        uploaded_file = st.file_uploader("OHLCV CSVをアップロード", type=["csv"], key="ohlcv_uploader")
        if uploaded_file is not None:
            try:
                raw_df = parse_csv(uploaded_file)
                st.success("CSVを読み込みました")
            except Exception as e:
                st.error(f"CSV読み込みエラー: {e}")
                raw_df = generate_sample_ohlcv()
        else:
            raw_df = generate_sample_ohlcv()
            st.info("CSV未選択のため、デモデータを表示しています")

    df = add_indicators(raw_df)
    last_close = float(df["Close"].iloc[-1])
    last_atr = float(df["ATR"].dropna().iloc[-1]) if df["ATR"].dropna().shape[0] else 0.0

    with st.expander("基本設定", expanded=True):
        symbol = st.text_input("銘柄名 / ティッカー", value="DEMO")
        calm_loss_budget = st.number_input("1回で平常心を保てる最大損失額", min_value=1.0, value=5000.0, step=500.0)
        atr_multiple = st.number_input("ATR係数（ノイズ耐性の確認用）", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slippage_factor = st.number_input("スリッページ係数", min_value=1.00, max_value=2.00, value=1.15, step=0.01)
        assumed_win_rate = st.slider("仮の勝率（期待値の点検用）", min_value=0.10, max_value=0.90, value=0.45, step=0.01)
        time_stop_bars = st.number_input("時間切れバー数", min_value=0, max_value=200, value=5, step=1)
        volume_profile_bins = st.slider("Volume Profile の価格ビン数", min_value=8, max_value=60, value=24, step=2)

    tabs = st.tabs(["プラン", "チャート", "ジャーナル"])

    with tabs[0]:
        st.subheader("売買プラン")
        entry_price = st.number_input("エントリー価格", min_value=0.0, value=round(last_close, 2), step=0.1)
        stop_price = st.number_input("ストップ価格", min_value=0.0, value=round(max(0.01, last_close - last_atr), 2), step=0.1)
        target_price = st.number_input("ターゲット価格", min_value=0.0, value=round(last_close + (2 * last_atr), 2), step=0.1)

        inferred_side = infer_side(entry_price, stop_price)
        default_index = 0 if inferred_side == "Long" else 1
        side = st.selectbox("売買方向", options=["Long", "Short"], index=default_index)

        risk = calc_risk(
            entry=entry_price,
            stop=stop_price,
            target=target_price,
            calm_loss_budget=calm_loss_budget,
            slippage_factor=slippage_factor,
        )

        rr = risk["rr_ratio"]
        expectancy_r = calc_expected_value(rr, assumed_win_rate)
        stop_vs_atr = (risk["structural_risk_per_share"] / max(last_atr, 1e-9)) if risk["structural_risk_per_share"] > 0 else 0.0

        c1, c2 = st.columns(2)
        c1.metric("推奨数量", f"{risk['quantity']}")
        c2.metric("実質RR", f"{rr:.2f}")

        c3, c4 = st.columns(2)
        c3.metric("想定損失総額", f"{risk['actual_risk_amount']:.2f}")
        c4.metric("想定利益総額", f"{risk['actual_reward_amount']:.2f}")

        c5, c6 = st.columns(2)
        c5.metric("最新ATR", f"{last_atr:.2f}")
        c6.metric("ストップ幅 ÷ ATR", f"{stop_vs_atr:.2f}")

        st.caption(f"期待値（簡易）: {expectancy_r:.2f}R / トレード")

        if risk["quantity"] <= 0:
            st.warning("現在の損失予算ではロットが0です。サイズを下げるか、エントリー/ストップを見直してください。")
        if rr < 1.2:
            st.warning("RRが低めです。『伸びる余地』が本当にあるか再確認してください。")
        if stop_vs_atr < atr_multiple:
            st.info("ストップ幅は設定したATR係数より狭いです。ノイズで狩られる可能性を再確認してください。")

        st.subheader("売買前ゲート")
        news_block = st.checkbox("重要イベント前後なので新規停止", value=False)
        volatility_block = st.checkbox("異常ボラティリティなので新規停止", value=False)
        revenge_block = st.checkbox("取り返しモードなので停止", value=False)
        sleep_block = st.checkbox("睡眠・体調不足なので停止", value=False)
        correlation_block = st.checkbox("相関資産が急変している", value=False)
        setup_grade = st.selectbox("セットアップ格付け", options=["A", "B", "C"], index=1)

        gate_passed, gate_reasons = gate_decision(
            news_block=news_block,
            volatility_block=volatility_block,
            revenge_block=revenge_block,
            sleep_block=sleep_block,
            correlation_block=correlation_block,
        )
        if gate_passed:
            st.success("ゲート通過: 新規エントリーを検討できる状態です。")
        else:
            st.error("ゲート不通過: " + " / ".join(gate_reasons))

        st.subheader("セットアップ監査")
        htf_aligned = st.checkbox("上位足の方向と一致", value=True)
        role_reversal_ok = st.checkbox("役割反転の再テスト良好", value=False)
        volume_quality_ok = st.checkbox("ブレイク時の出来高の質が良い", value=False)
        room_to_run_ok = st.checkbox("上に/下に真空地帯がある", value=False)

        setup_quality_score = setup_score(
            htf_aligned=htf_aligned,
            role_reversal_ok=role_reversal_ok,
            volume_quality_ok=volume_quality_ok,
            room_to_run_ok=room_to_run_ok,
            setup_grade=setup_grade,
        )

        d1, d2 = st.columns(2)
        d1.metric("セットアップ品質", f"{setup_quality_score}/100")
        d2.metric("100回やりたい形", "YES" if setup_quality_score >= 70 and gate_passed else "NO")

        if setup_quality_score < 70:
            st.warning("品質スコアが低めです。『入れる』より『伸ばせる』を優先してください。")

        st.subheader("時間切れ撤退")
        st.metric("設定中の時間切れバー数", f"{time_stop_bars}")
        time_stop_triggered = st.checkbox("今回、時間切れ撤退に該当した", value=False)
        st.caption("チャートのオレンジ縦線までに進展がなければ、建値付近撤退を検討")

        st.subheader("監査メモ")
        rule_adherence_score = st.slider("ルール遵守度", min_value=0, max_value=100, value=100, step=5)
        would_repeat_100x = st.radio("このトレードを100回繰り返したいですか？", options=["Yes", "No"], horizontal=True)
        notes = st.text_area(
            "メモ",
            value="見送り判断 / 早逃げ判断 / 外部環境 / 心理状態 / 良かった点 / 改善点",
            height=160,
        )

        if st.button("このトレード監査を保存", type="primary"):
            record = TradeRecord(
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol=symbol,
                side=side,
                entry_price=float(entry_price),
                stop_price=float(stop_price),
                target_price=float(target_price),
                structural_risk_per_share=float(risk["structural_risk_per_share"]),
                effective_risk_per_share=float(risk["effective_risk_per_share"]),
                reward_per_share=float(risk["reward_per_share"]),
                rr_ratio=float(risk["rr_ratio"]),
                quantity=int(risk["quantity"]),
                calm_loss_budget=float(calm_loss_budget),
                atr=float(last_atr),
                atr_multiple=float(atr_multiple),
                slippage_factor=float(slippage_factor),
                setup_grade=setup_grade,
                gate_passed=int(gate_passed),
                news_block=int(news_block),
                volatility_block=int(volatility_block),
                revenge_block=int(revenge_block),
                sleep_block=int(sleep_block),
                correlation_block=int(correlation_block),
                htf_aligned=int(htf_aligned),
                role_reversal_ok=int(role_reversal_ok),
                volume_quality_ok=int(volume_quality_ok),
                room_to_run_ok=int(room_to_run_ok),
                time_stop_bars=int(time_stop_bars),
                time_stop_triggered=int(time_stop_triggered),
                rule_adherence_score=int(rule_adherence_score),
                would_repeat_100x=would_repeat_100x,
                notes=notes,
            )
            save_trade(record)
            st.success("ジャーナルに保存しました。")
            st.cache_data.clear()

    with tabs[1]:
        st.subheader("チャート確認")
        fig = build_chart(
            df.tail(180),
            entry_price,
            stop_price,
            target_price,
            time_stop_bars=time_stop_bars,
            volume_profile_bins=volume_profile_bins,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("右側に簡易Volume Profile、オレンジ縦線にTime Stopを表示")

    with tabs[2]:
        st.subheader("ジャーナル")
        journal_df = load_journal()
        if journal_df.empty:
            st.info("まだ記録がありません。1件保存するとここに表示されます。")
        else:
            c1, c2 = st.columns(2)
            c1.metric("記録数", f"{len(journal_df)}")
            c2.metric("平均遵守度", f"{journal_df['rule_adherence_score'].mean():.1f}")

            discipline_df = journal_df.copy().sort_values("created_at").tail(20)
            discipline_df["repeat_yes"] = discipline_df["would_repeat_100x"].eq("Yes").astype(int) * 100
            discipline_df["repeat_yes_cum"] = discipline_df["repeat_yes"].expanding().mean()
            discipline_df["discipline_curve"] = discipline_df["rule_adherence_score"].expanding().mean()

            discipline_fig = go.Figure()
            discipline_fig.add_scatter(
                x=discipline_df["created_at"],
                y=discipline_df["rule_adherence_score"],
                mode="lines+markers",
                name="Rule Adherence",
            )
            discipline_fig.add_scatter(
                x=discipline_df["created_at"],
                y=discipline_df["repeat_yes_cum"],
                mode="lines+markers",
                name="Would Repeat Yes %",
            )
            discipline_fig.add_scatter(
                x=discipline_df["created_at"],
                y=discipline_df["discipline_curve"],
                mode="lines",
                name="Discipline Curve",
            )
            discipline_fig.update_layout(
                height=360,
                yaxis_title="Score / Rate (%)",
                xaxis_title="Recorded At",
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(discipline_fig, use_container_width=True)
            st.dataframe(journal_df, use_container_width=True, hide_index=True)

            csv = journal_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="ジャーナルCSVをダウンロード",
                data=csv,
                file_name="trade_journal_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with st.expander("この版でやっていること / まだやっていないこと", expanded=False):
        st.markdown(
            """
            **やっていること**
            - スマホ向けの縦スクロールUI
            - 売買前ゲートで見送り条件を強制
            - 平常心ベースの損失額から数量を逆算
            - スリッページ込みRR計算
            - Time Stop の縦線表示
            - 簡易Volume Profile と POC 表示
            - 規律曲線のダッシュボード

            **まだやっていないこと**
            - 自動売買 / ブローカー発注
            - 市場全体スキャン
            - 複数時間足の自動判定
            - 本格的な VAH / VAL 計算
            - バックテスト一体化
            """
        )


if __name__ == "__main__":
    main()
