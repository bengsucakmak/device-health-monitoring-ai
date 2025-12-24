from __future__ import annotations
from matplotlib import font_manager

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Matplotlib (for embedded images)
import matplotlib.pyplot as plt


@dataclass
class ReportMeta:
    threshold: float
    method: str
    lookback_hours: int
    alert_window_hours: int
    health_score: float
    status_label: str

    # Optional extras
    last_anomaly: str | None = None
    repeats_24h: int | None = None
    consecutive_windows: int | None = None

    # Productization
    product_name: str = "Cihaz Sağlığı Asistanı"
    customer_label: str | None = None
    recommendation_title: str | None = None
    recommendation_bullets: list[str] | None = None
    report_date: str | None = None


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _save_lineplot_png(
    out_png: Path,
    x: pd.Series,
    y: pd.Series,
    title: str,
    ylabel: str,
    hline: float | None = None,
) -> None:
    _ensure_parent(out_png)

    plt.figure(figsize=(10, 2.8), dpi=160)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.plot(x, y)
    if hline is not None:
        plt.axhline(hline, linestyle="--")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def export_pdf_report(
    out_path: str,
    meta: ReportMeta,
    pred_view: pd.DataFrame,
    scores_view: pd.DataFrame,
) -> str:
    """
    Customer-ready PDF with embedded charts + TR character support via TTF fonts.
    pred_view columns: ['timestamp','predicted_power']
    scores_view columns: ['timestamp','anomaly_score']
    """
    out_pdf = Path(out_path)
    _ensure_parent(out_pdf)

    # temp images
    tmp_dir = out_pdf.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Prepare health series
    s = scores_view.sort_values("timestamp").copy()
    eps = 1e-12
    ratio = (s["anomaly_score"].to_numpy(dtype=np.float32) + eps) / float(meta.threshold + eps)
    health = 100.0 - np.clip((ratio - 1.0) * 60.0, 0.0, 100.0)
    s["health"] = health

    # ---- Save charts
    png_score = tmp_dir / "anomaly_score.png"
    _save_lineplot_png(
        png_score,
        x=s["timestamp"],
        y=s["anomaly_score"],
        title="Anomali Skoru (threshold çizgisi ile)",
        ylabel="Skor",
        hline=float(meta.threshold),
    )

    png_health = tmp_dir / "health_trend.png"
    _save_lineplot_png(
        png_health,
        x=s["timestamp"],
        y=s["health"],
        title="Sağlık Trendi (0–100)",
        ylabel="Sağlık",
        hline=None,
    )

    p = pred_view.sort_values("timestamp").copy()
    png_power = tmp_dir / "fridge_power.png"
    _save_lineplot_png(
        png_power,
        x=p["timestamp"],
        y=p["predicted_power"],
        title="Buzdolabı Tahmini Güç",
        ylabel="W",
        hline=None,
    )

    # ----------------------------
    # Build PDF
    # ----------------------------
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4

    # --- Unicode (TR) font setup (NO local font files needed) ---
    def _register_tr_fonts():
        """
        Tries to register DejaVu Sans from Matplotlib/system.
        This usually exists even if you don't ship any font files.
        """
        base_font = "Helvetica"
        bold_font = "Helvetica-Bold"

        try:
            # Matplotlib almost always ships DejaVu Sans
            base_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
            pdfmetrics.registerFont(TTFont("DejaVu", base_path))
            base_font = "DejaVu"

            # Try bold variant (best effort)
            try:
                bold_path = font_manager.findfont("DejaVu Sans:style=Bold", fallback_to_default=False)
                pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold_path))
                bold_font = "DejaVu-Bold"
            except Exception:
                bold_font = "DejaVu"  # bold yoksa aynı fontla devam (TR çalışır)

        except Exception:
            # If everything fails, fall back (TR may break on Helvetica)
            base_font = "Helvetica"
            bold_font = "Helvetica-Bold"

        return base_font, bold_font


    base_font, bold_font = _register_tr_fonts()

    
    def set_title(size: int = 18):
        c.setFont(bold_font, size)

    def set_text(size: int = 10):
        c.setFont(base_font, size)

    def footer(beta: str = "Beta Demo"):
        set_text(9)
        c.drawString(2.0 * cm, 1.6 * cm,
                     "Not: Bu rapor teşhis değildir; erken uyarı amaçlıdır. Yanlış alarm riski olabilir.")
        c.drawRightString(w - 2.0 * cm, 1.6 * cm, beta)

    def cover_page():
        set_title(22)
        c.drawString(2.0 * cm, h - 2.6 * cm, meta.product_name)

        set_text(12)
        date_txt = meta.report_date or ""
        c.drawString(2.0 * cm, h - 3.4 * cm, f"Rapor Tarihi: {date_txt}")
        if meta.customer_label:
            c.drawString(2.0 * cm, h - 4.0 * cm, f"Kullanıcı / Cihaz: {meta.customer_label}")

        set_title(18)
        c.drawString(2.0 * cm, h - 5.3 * cm, f"Sağlık: {meta.health_score:.0f}/100")

        set_title(14)
        c.drawString(2.0 * cm, h - 6.1 * cm, f"Durum: {meta.status_label}")

        footer()

    def executive_summary():
        set_title(16)
        c.drawString(2.0 * cm, h - 2.6 * cm, "Özet (Executive Summary)")

        set_text(11)
        y = h - 3.6 * cm
        if meta.repeats_24h is not None:
            c.drawString(2.0 * cm, y, f"- Son 24 saatte olay sayısı: {meta.repeats_24h}")
            y -= 0.6 * cm
        if meta.consecutive_windows is not None:
            c.drawString(2.0 * cm, y, f"- Art arda anomali penceresi: {meta.consecutive_windows}")
            y -= 0.6 * cm
        if meta.last_anomaly:
            c.drawString(2.0 * cm, y, f"- Son anomali zamanı: {meta.last_anomaly}")
            y -= 0.6 * cm

        if meta.recommendation_title:
            y -= 0.2 * cm
            set_title(12)
            c.drawString(2.0 * cm, y, "Öneri")
            y -= 0.6 * cm

            set_text(11)
            c.drawString(2.0 * cm, y, meta.recommendation_title)
            y -= 0.75 * cm

            if meta.recommendation_bullets:
                for b in meta.recommendation_bullets[:10]:
                    c.drawString(2.3 * cm, y, f"• {b}")
                    y -= 0.55 * cm
                    if y < 3.0 * cm:
                        break

        footer()

    def draw_img(path: Path, x_cm: float, y_cm: float, width_cm: float) -> float:
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        aspect = ih / float(iw)
        width = width_cm * cm
        height = width * aspect
        c.drawImage(img, x_cm * cm, y_cm * cm, width=width, height=height,
                    preserveAspectRatio=True, mask="auto")
        return height / cm

    def charts_page():
        # Header
        set_title(18)
        c.drawString(2.0 * cm, h - 2.3 * cm, "Cihaz Sağlığı Raporu (Buzdolabı)")

        set_text(10)
        c.drawString(2.0 * cm, h - 3.0 * cm, "Bu rapor erken uyarı amaçlıdır; teşhis niteliği taşımaz.")

        # Meta line
        set_title(14)
        c.drawString(2.0 * cm, h - 3.9 * cm,
                     f"Sağlık: {meta.health_score:.0f}/100  |  Durum: {meta.status_label}")

        set_text(10)
        c.drawString(2.0 * cm, h - 4.6 * cm,
                     f"Threshold: {meta.threshold:.6f}  (Kaynak: {meta.method})")
        c.drawString(2.0 * cm, h - 5.1 * cm,
                     f"Rapor penceresi: Son {meta.lookback_hours} saat  |  Alarm penceresi: {meta.alert_window_hours} saat")

        # Optional incident facts
        y0 = h - 5.7 * cm
        if meta.last_anomaly or (meta.repeats_24h is not None) or (meta.consecutive_windows is not None):
            set_title(11)
            c.drawString(2.0 * cm, y0, "Son Olay Özeti")
            set_text(10)
            y0 -= 0.55 * cm
            if meta.last_anomaly:
                c.drawString(2.0 * cm, y0, f"- Son anomali zamanı: {meta.last_anomaly}")
                y0 -= 0.45 * cm
            if meta.repeats_24h is not None:
                c.drawString(2.0 * cm, y0, f"- Son 24 saatte olay sayısı: {meta.repeats_24h}")
                y0 -= 0.45 * cm
            if meta.consecutive_windows is not None:
                c.drawString(2.0 * cm, y0, f"- Art arda anomali penceresi: {meta.consecutive_windows}")
                y0 -= 0.45 * cm

        # Charts stacked
        y_cursor_cm = 18.2 - 7.0
        set_title(12)
        c.drawString(2.0 * cm, (y_cursor_cm + 0.6) * cm, "Zaman Serileri")
        y_cursor_cm -= 0.2

        h1 = draw_img(png_score, x_cm=2.0, y_cm=y_cursor_cm, width_cm=17.0)
        y_cursor_cm -= (h1 + 0.6)

        h2 = draw_img(png_health, x_cm=2.0, y_cm=y_cursor_cm, width_cm=17.0)
        y_cursor_cm -= (h2 + 0.6)

        if y_cursor_cm < 4.0:
            footer()
            c.showPage()
            y_cursor_cm = 24.5
            set_title(12)
            c.drawString(2.0 * cm, (y_cursor_cm - 1.0) * cm, "Zaman Serileri (devam)")
            y_cursor_cm -= 1.6

        _ = draw_img(png_power, x_cm=2.0, y_cm=y_cursor_cm, width_cm=17.0)

        footer()

    # --- Page order ---
    cover_page()
    c.showPage()

    executive_summary()
    c.showPage()

    charts_page()
    c.save()

    return str(out_pdf)
