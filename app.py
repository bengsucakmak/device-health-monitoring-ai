from __future__ import annotations
from src.pipeline.alerts import build_alert_summary
from src.pipeline.trends import health_series_from_scores, health_delta_last_hours
from src.pipeline.usage_insights import hourly_event_profile


import sys
import json
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is in sys.path so "import src..." works
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.pipeline.health import health_score, health_band
from src.pipeline.pdf_report import ReportMeta, export_pdf_report

# ------------------------------------------------------------
# Data paths
# ------------------------------------------------------------
DATA_PRED = Path("data/processed/fridge_pred_dedup.csv")
DATA_SCORES = Path("data/processed/anomaly_scores.csv")
DATA_THRESHOLD = Path("data/processed/threshold.json")


# ------------------------------------------------------------
# Styling (B2C feel)
# ------------------------------------------------------------
B2C_CSS = """
<style>
/* tighter top padding */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* metric cards look nicer */
div[data-testid="metric-container"] {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  background: rgba(255,255,255,0.03);
}
/* HERO */
.hero {
  border-radius: 22px;
  padding: 18px 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
.hero.green { background: linear-gradient(135deg, rgba(46,204,113,0.18), rgba(255,255,255,0.03)); }
.hero.orange { background: linear-gradient(135deg, rgba(243,156,18,0.18), rgba(255,255,255,0.03)); }
.hero.red { background: linear-gradient(135deg, rgba(231,76,60,0.18), rgba(255,255,255,0.03)); }

.hero-top {
  display:flex; align-items:center; justify-content:space-between; gap:12px;
}
.badge {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  font-weight:700; font-size:0.85rem;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(0,0,0,0.15);
}
.badge.beta { border-color: rgba(155,89,182,0.45); }
.hero-score {
  font-size: 3.2rem;
  font-weight: 800;
  line-height: 1.0;
  margin-top: 10px;
}
.hero-sub {
  color: rgba(255,255,255,0.75);
  font-size: 1.0rem;
  margin-top: 6px;
}
.kv {
  display:flex; gap:16px; flex-wrap:wrap;
  margin-top: 12px;
}
.kv .item {
  padding:10px 12px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  min-width: 210px;
}
.kv .k { font-size:0.82rem; color: rgba(255,255,255,0.70); }
.kv .v { font-size:1.05rem; font-weight:700; margin-top:4px; }

/* status pill */
.pill {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:8px 12px;
  border-radius:999px;
  font-weight:600;
  font-size:0.95rem;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
}
.pill.green { border-color: rgba(46, 204, 113, 0.35); }
.pill.orange { border-color: rgba(243, 156, 18, 0.35); }
.pill.red { border-color: rgba(231, 76, 60, 0.35); }

.subtle {
  color: rgba(255,255,255,0.70);
  font-size: 0.95rem;
}
.small {
  color: rgba(255,255,255,0.65);
  font-size: 0.85rem;
}
.hr {
  margin: 0.6rem 0 1.0rem 0;
  border-top: 1px solid rgba(255,255,255,0.10);
}
</style>
"""


# ------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pred() -> pd.DataFrame:
    if not DATA_PRED.exists():
        raise FileNotFoundError(
            "data/processed/fridge_pred_dedup.csv bulunamadı.\n"
            "Önce: 03_infer_model1 + 03b_deduplicate_fridge_csv çalıştır."
        )
    df = pd.read_csv(DATA_PRED, parse_dates=["timestamp"]).sort_values("timestamp")
    if "predicted_power" not in df.columns:
        raise ValueError("fridge_pred_dedup.csv içinde 'predicted_power' kolonu yok.")
    return df


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    if not DATA_SCORES.exists():
        raise FileNotFoundError(
            "data/processed/anomaly_scores.csv bulunamadı.\n"
            "Önce: 05_score_anomalies çalıştır."
        )
    df = pd.read_csv(DATA_SCORES, parse_dates=["timestamp"]).sort_values("timestamp")
    if "anomaly_score" not in df.columns:
        raise ValueError("anomaly_scores.csv içinde 'anomaly_score' kolonu yok.")
    return df


def load_threshold(scores_df: pd.DataFrame) -> tuple[float, str]:
    # Preferred: threshold.json
    if DATA_THRESHOLD.exists():
        info = json.loads(DATA_THRESHOLD.read_text(encoding="utf-8"))
        return float(info["threshold"]), str(info.get("method", "threshold.json"))

    # Fallback: estimate
    if "is_anomaly" in scores_df.columns and scores_df["is_anomaly"].astype(bool).any():
        th = float(scores_df.loc[scores_df["is_anomaly"].astype(bool), "anomaly_score"].min())
        return th, "estimated(min anomaly score)"
    th = float(np.percentile(scores_df["anomaly_score"].values, 99.0))
    return th, "estimated(P99)"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def pill_html(label: str, emoji: str) -> str:
    cls = "green" if "Healthy" in label else "orange" if "Watch" in label else "red"
    return f'<span class="pill {cls}">{emoji} {label}</span>'

def hero_html(hs: float, label: str, emoji: str, alert_severity: str) -> str:
    cls = "green" if label == "Healthy" else "orange" if label == "Watch" else "red"
    sev_txt = "Normal İzleme" if alert_severity == "none" else ("Soft Uyarı" if alert_severity == "soft" else "Kritik Uyarı")
    return f"""
    <div class="hero {cls}">
      <div class="hero-top">
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <span class="badge beta">🧪 Beta Demo</span>
          <span class="pill {cls}">{emoji} {label}</span>
          <span class="badge">🔔 {sev_txt}</span>
        </div>
        <div class="small">Bu bir teşhis değil, erken uyarı sistemidir.</div>
      </div>

      <div class="hero-score">{hs:.0f}/100</div>
      <div class="hero-sub">Cihaz sağlığı tek bakışta. Anomali skoruna göre hesaplanır.</div>
    </div>
    """


def make_health_series(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    s = scores_df.copy()
    hs = health_score(s["anomaly_score"].to_numpy(dtype=np.float32), threshold)
    s["health"] = hs
    return s

def top_recommendation_v2(label: str, alert_severity: str) -> tuple[str, str, list[str]]:
    """
    Returns: (title, body, bullets)
    """
    if label == "Healthy" and alert_severity == "none":
        return (
            "Her şey yolunda görünüyor.",
            "Cihaz davranışı olağan aralıkta. Yine de küçük önlemlerle performansı koruyabilirsin.",
            [
                "Kapı contası tam kapanıyor mu kontrol et",
                "Arka havalandırma boşluklarını kapatma",
                "Aşırı doluluk ve sıcak yiyecek koyma gibi durumları azalt",
            ],
        )

    if label == "Watch" or alert_severity == "soft":
        return (
            "Sınırda dalgalanma var — bugün kontrol önerilir.",
            "Bu genelde çevresel koşullar veya kullanım alışkanlıkları kaynaklı olabilir. 24 saat daha izleyelim.",
            [
                "Kapı açık kalma / sık aç-kapa var mı?",
                "Ortam sıcaklığı arttı mı? (mutfak/ yaz etkisi)",
                "Buzlanma/ aşırı soğutma belirtisi var mı?",
            ],
        )

    return (
        "Anomali tekrarlıyor — fiziksel kontrol önerilir.",
        "Bu bir teşhis değildir; ancak tekrar eden anomali sinyali bakım ihtimalini artırır.",
        [
            "Anormal ses/titreşim ve aşırı ısınma var mı kontrol et",
            "Kompresör çok sık devreye giriyor mu gözle",
            "Devam ederse servis/teknik kontrol planla",
        ],
    )



def fig_line(df: pd.DataFrame, x: str, y: str, title: str, threshold: float | None = None):
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=340)
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash")
    return fig


def fig_sparkline(y: np.ndarray, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines"))
    fig.update_layout(
        title=title,
        height=120,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Cihaz Sağlığı Asistanı", layout="wide")
    st.markdown(B2C_CSS, unsafe_allow_html=True)

    # Header
    left_h, right_h = st.columns([0.72, 0.28])
    with left_h:
        st.title("🧊 Cihaz Sağlığı Asistanı")
        st.markdown('<div class="subtle">Buzdolabı tüketiminden “sağlık skoru” ve anomali uyarısı üreten B2C demo paneli.</div>', unsafe_allow_html=True)
    with right_h:
        st.markdown('<div class="small"><span class="badge beta">🧪 Beta Demo</span></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="small">Model-1: Disaggregation (TCN)<br/>Model-2: LSTM Autoencoder</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="small">⚠️ Bu bir teşhis değil, erken uyarı sistemidir.<br/>Yanlış alarm riski olabilir.</div>',
            unsafe_allow_html=True,
        )

    # Load data
    try:
        pred = load_pred()
        scores = load_scores()
    except Exception as e:
        st.error(str(e))
        st.stop()

    threshold, th_method = load_threshold(scores)

    # Sidebar controls
    st.sidebar.header("Filtreler")
    presets = st.sidebar.selectbox("Zaman aralığı", ["Son 24 saat", "Son 48 saat", "Son 7 gün", "Özel"], index=1)
    if presets == "Son 24 saat":
        lookback_hours = 24
    elif presets == "Son 48 saat":
        lookback_hours = 48
    elif presets == "Son 7 gün":
        lookback_hours = 168
    else:
        lookback_hours = st.sidebar.slider("Son kaç saat?", 6, 720, 72, step=6)

    alert_window_hours = st.sidebar.slider("Uyarı penceresi (saat)", 1, 72, 12, step=1)
    smooth_scores = st.sidebar.checkbox("Skoru yumuşat (rolling mean)", value=True)
    show_tables = st.sidebar.checkbox("Ham tabloları göster", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Demo Simülasyon (What-if)")
    sim_pct = st.sidebar.slider("Threshold değişimi (%)", -20, 20, 0, 5)
    sim_threshold = threshold * (1 + sim_pct / 100.0)


    # Time window for display
    end_time = scores["timestamp"].max()
    start_time = end_time - pd.Timedelta(hours=int(lookback_hours))

    pred_view = pred[(pred["timestamp"] >= start_time) & (pred["timestamp"] <= end_time)].copy()
    scores_view = scores[(scores["timestamp"] >= start_time) & (scores["timestamp"] <= end_time)].copy()

    if smooth_scores and len(scores_view) > 5:
        scores_view["anomaly_score_raw"] = scores_view["anomaly_score"]
        scores_view["anomaly_score"] = scores_view["anomaly_score"].rolling(5, min_periods=1).mean()

    # Recent window for alert/health
    recent_scores = scores[scores["timestamp"] >= (end_time - pd.Timedelta(hours=int(alert_window_hours)))]
    if len(recent_scores) == 0:
        recent_scores = scores.tail(50)

    recent_mean_score = float(recent_scores["anomaly_score"].mean())
    hs = float(health_score(np.array([recent_mean_score], dtype=np.float32), threshold=threshold)[0])
    label, emoji = health_band(hs)
    recent_anom = bool((recent_scores["anomaly_score"] > threshold).any())
    health_df = health_series_from_scores(scores, threshold)
    delta_24h = health_delta_last_hours(health_df, hours=24)
    last_update_ts = pd.Timestamp(end_time).strftime("%Y-%m-%d %H:%M")

    alert_sum = build_alert_summary(scores_df=scores, threshold=sim_threshold, now=end_time, lookback_hours=24)


    rec_title, rec_body, rec_bullets = top_recommendation_v2(label, alert_sum.severity)

    

    st.markdown(hero_html(hs, label, emoji, alert_sum.severity), unsafe_allow_html=True)
    trend_txt = "iyileşti ✅" if delta_24h > 2 else ("kötüleşti ⚠️" if delta_24h < -2 else "stabil 🟰")
    st.caption(f"🕒 Son güncelleme: **{last_update_ts}**  |  24s trend: **{delta_24h:+.1f} puan** ({trend_txt})")


    # Trust / info box (küçük, güven veren)
    st.info(
        "Bu panel **erken uyarı** amaçlıdır. Zaman zaman **yanlış alarm** üretebilir. "
        "Karar vermeden önce fiziksel gözlem ve gerekirse teknik servis ile doğrulayın."
    )

    with st.expander("📈 Health Trend (son 72 saat)", expanded=False):
        view_hrs = 72
        tmin = pd.Timestamp(end_time) - pd.Timedelta(hours=view_hrs)
        spark = health_df[health_df["timestamp"] >= tmin].copy()
        spark = spark.set_index("timestamp")[["health"]]
        st.line_chart(spark)


    # KPI row
    c1, c2, c3, c4, c5 = st.columns([0.22, 0.20, 0.20, 0.18, 0.20])
    c1.metric("Sağlık Skoru", f"{hs:.0f}/100")
    c2.metric("Son 24s Olay Sayısı", f"{alert_sum.repeats_24h}")
    c3.metric("Art Arda Anomali", f"{alert_sum.consecutive_windows} pencere")
    c4.metric("Threshold (sim)", f"{sim_threshold:.6f}")
    last_ts = "-" if alert_sum.last_anomaly_ts is None else alert_sum.last_anomaly_ts.strftime("%Y-%m-%d %H:%M")
    c5.metric("Son Anomali", last_ts)

    st.caption("ℹ️ *Olay sayısı*, birbirine yakın anomali noktalarının tek bir olay olarak kümelenmiş halidir (spam bildirim önlemek için).")
    
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Recommendation card
    st.subheader("✅ Ne Yapmalıyım?")
    bullets_md = "\n".join([f"- {b}" for b in rec_bullets])

    if alert_sum.severity == "none" and label == "Healthy":
        st.success(f"**{rec_title}**\n\n{rec_body}\n\n{bullets_md}")
    elif alert_sum.severity in ["soft"] or label == "Watch":
        st.warning(f"**{rec_title}**\n\n{rec_body}\n\n{bullets_md}")
    else:
        st.error(f"**{rec_title}**\n\n{rec_body}\n\n{bullets_md}")

    with st.expander("🧾 Benim için özetle", expanded=False):
        last_txt = "yok" if alert_sum.last_anomaly_ts is None else alert_sum.last_anomaly_ts.strftime("%Y-%m-%d %H:%M")
        st.write(
            f"1) Şu anki sağlık durumu: **{hs:.0f}/100 ({label})**.\n\n"
            f"2) Son 24 saatte **{alert_sum.repeats_24h} olay** görüldü; son anomali: **{last_txt}**.\n\n"
            f"3) Öneri: **{rec_title}** — {', '.join(rec_bullets[:2])}."
        )


    # Tabs for B2C navigation
    tab_overview, tab_trends, tab_alerts, tab_report = st.tabs(["🏠 Genel Bakış", "📈 Trendler", "🚨 Uyarılar", "📄 Rapor"])

    with tab_overview:
        left, right = st.columns(2)

        with left:
            st.subheader("🧊 Fridge Güç Tahmini")
            st.caption("Model-1 çıktısı (dedup). Kullanıcıya cihaz davranışını görünür kılar.")
            fig1 = fig_line(pred_view, "timestamp", "predicted_power", "Predicted Fridge Power (Dedup)")
            st.plotly_chart(fig1, use_container_width=True)

        with right:
            st.subheader("📉 Anomali Skoru")
            st.caption("Model-2 reconstruction error. Threshold üstü: anomali.")
            fig2 = fig_line(scores_view, "timestamp", "anomaly_score", "Anomaly Score", threshold=threshold)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_trends:
        st.subheader("🟣 Health Score Trend")
        st.caption("Skor → Health(0-100) dönüşümü. Kullanıcı diline çevrilmiş hali.")
        health_df = make_health_series(scores_view, threshold)
        fig_h = px.line(health_df, x="timestamp", y="health", title="Health Score Over Time (0–100)")
        fig_h.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=340)
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("### ⚡ Mini Özet")
        col_a, col_b = st.columns(2)
        with col_a:
            last_n = min(200, len(scores_view))
            st.plotly_chart(fig_sparkline(scores_view["anomaly_score"].tail(last_n).to_numpy(), "Skor Sparkline"), use_container_width=True)
        with col_b:
            last_n2 = min(200, len(health_df))
            st.plotly_chart(fig_sparkline(health_df["health"].tail(last_n2).to_numpy(), "Health Sparkline"), use_container_width=True)

        st.subheader("🧭 Kullanım Bağlamı (Saatlik Yoğunluk)")
        hp = hourly_event_profile(scores, threshold).set_index("hour")
        st.bar_chart(hp)
        st.caption("Bu grafik, günün hangi saatlerinde anomali noktalarının daha yoğun olduğunu gösterir. Tek başına teşhis değildir.")


    with tab_alerts:
        st.subheader("🚨 Anomali Olayları")
        st.caption("Threshold aşan noktaları listeler. B2C için ‘son olaylar’ mantığı.")

        st.markdown("### 🔎 Son Olay Özeti")
        colx1, colx2, colx3 = st.columns(3)
        colx1.metric("Son 24s Olay", f"{alert_sum.repeats_24h}")
        colx2.metric("Art Arda Anomali", f"{alert_sum.consecutive_windows} pencere")
        lt = "-" if alert_sum.last_anomaly_ts is None else alert_sum.last_anomaly_ts.strftime("%Y-%m-%d %H:%M")
        colx3.metric("Son Anomali", lt)

        if alert_sum.severity == "critical":
            st.error("🚨 Kritik uyarı: Tekrarlayan/uzayan anomali sinyali.")
        elif alert_sum.severity == "soft":
            st.warning("🔔 Soft uyarı: Sınırda dalgalanma. Bugün kontrol + izleme önerilir.")
        else:
            st.success("✅ Son 24 saatte belirgin bir anomali olayı yok.")

        view_flags = scores_view.copy()
        view_flags["is_anomaly"] = view_flags["anomaly_score"] > threshold
        anom_df = view_flags[view_flags["is_anomaly"]].copy()

        # scatter
        fig3 = px.scatter(
            view_flags,
            x="timestamp",
            y="anomaly_score",
            color="is_anomaly",
            title="Anomaly Points (threshold-based)",
        )
        fig3.add_hline(y=threshold, line_dash="dash")
        fig3.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=340)
        st.plotly_chart(fig3, use_container_width=True)

        if len(anom_df) == 0:
            st.success("Seçili aralıkta threshold aşımı yok.")
        else:
            st.warning(f"Seçili aralıkta **{len(anom_df)}** adet anomali noktası var.")
            # Son 25 olayı göster
            anom_df = anom_df.sort_values("timestamp", ascending=False).head(25)
            anom_df["timestamp"] = anom_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(anom_df[["timestamp", "anomaly_score"]], use_container_width=True, hide_index=True)

    with tab_report:
        st.subheader("📄 PDF Rapor")
        st.caption("Seçili zaman penceresinin özet istatistiklerini PDF olarak indir.")

        st.info(
            "Not: Bu PDF özet rapordur (istatistik). İstersen bir sonraki adımda dashboard grafiklerini de PDF’e gömebiliriz."
        )

        col_r1, col_r2 = st.columns([0.55, 0.45])
        with col_r1:
            st.write("**Rapor içeriği**")
            st.markdown(
                "- Threshold ve yöntemi\n"
                "- Sağlık skoru ve durum bandı\n"
                "- Seçili zaman aralığında skor istatistikleri\n"
                "- Seçili zaman aralığında güç istatistikleri\n"
            )

        with col_r2:
            if st.button("📄 PDF Oluştur(Müşteriye göndermek için)", use_container_width=True):
                last_ts = None if alert_sum.last_anomaly_ts is None else alert_sum.last_anomaly_ts.strftime("%Y-%m-%d %H:%M")
                meta = ReportMeta(
                    threshold=threshold,
                    method=th_method,
                    lookback_hours=int(lookback_hours),
                    alert_window_hours=int(alert_window_hours),
                    health_score=float(hs),
                    status_label=f"{emoji} {label}",
                    last_anomaly=last_ts,
                    repeats_24h=int(alert_sum.repeats_24h),
                    consecutive_windows=int(alert_sum.consecutive_windows),
                    report_date=pd.Timestamp(end_time).strftime("%Y-%m-%d %H:%M"),
                    customer_label="Demo Home-Fridge",  # seçili ev/cihaz etiketi gibi kullan
                    recommendation_title=rec_title,
                    recommendation_bullets=rec_bullets,
                
                )

                pdf_path = export_pdf_report(
                    out_path="outputs/reports/fridge_health_report.pdf",
                    meta=meta,
                    pred_view=pred_view,
                    scores_view=scores_view,
                )
                st.success("✅ PDF oluşturuldu.")
                st.download_button(
                    label="PDF indir",
                    data=Path(pdf_path).read_bytes(),
                    file_name="fridge_health_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    if show_tables:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("Ham Veri (Debug)")
        st.write("pred_view (tail)", pred_view.tail(200))
        st.write("scores_view (tail)", scores_view.tail(200))


if __name__ == "__main__":
    main()
