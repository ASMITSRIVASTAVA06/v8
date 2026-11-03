"""
TerraVision AI — Alerts section.

Browse every alert, filter and select rows, record the review outcome
(status / assignee / note / confirmed type), see exactly where each alert is,
then download or share them (CSV, JSON, printable report, e-mail, Telegram,
WhatsApp). Reviewed alerts can be exported as training labels for the model.
"""

from __future__ import annotations

from typing import Dict, List
from urllib.parse import quote

import pandas as pd
import streamlit as st

import store
from alert_tools import (
    CATEGORY_STYLE,
    LIFECYCLES,
    MODEL_CLASSES,
    SEVERITY_COLOR,
    SEVERITY_ORDER,
    VERDICT_ORDER,
    build_html_report,
    build_summary_text,
    build_training_csv,
    category_label,
    counts_by,
    get_smtp_config,
    get_telegram_config,
    gmail_link,
    mailto_link,
    parse_recipients,
    send_alert_email,
    send_telegram,
    to_csv_bytes,
    to_dataframe,
    to_json_bytes,
)
from locator import render_location_explorer


def _wide_kwargs() -> Dict:
    """`width="stretch"` on newer Streamlit, `use_container_width=True` on older releases."""
    import inspect
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


WIDE = _wide_kwargs()

FORMATS = {
    "CSV (spreadsheet)": ("terravision_alerts.csv", "text/csv"),
    "JSON (full detail)": ("terravision_alerts.json", "application/json"),
    "HTML report (print to PDF)": ("terravision_alert_report.html", "text/html"),
}


def _severity_strip(df: pd.DataFrame) -> None:
    counts = counts_by(df, "severity", SEVERITY_ORDER)
    if not counts:
        return
    import theme
    st.markdown(theme.kpi_html([(v, k, SEVERITY_COLOR[k]) for k, v in counts.items()], min_width=130),
                unsafe_allow_html=True)


def _filters(df: pd.DataFrame, ns: str, context_ok: List[str]) -> pd.DataFrame:
    c1, c2, c3 = st.columns([2, 2, 2])
    cats = [c for c in CATEGORY_STYLE if c in set(df["category"])]
    cats += [c for c in sorted(set(df["category"])) if c not in cats]
    sel_cats = c1.multiselect("Alert type", cats, default=cats,
                              format_func=category_label, key=f"al_{ns}_cats")
    sevs = [s for s in SEVERITY_ORDER if s in set(df["severity"])]
    sel_sevs = c2.multiselect("Severity", sevs, default=sevs, key=f"al_{ns}_sevs")
    search = c3.text_input("Search (ID, region, assignee, note)", key=f"al_{ns}_search",
                           placeholder="e.g. TV-3F2A, Ramesh, 27.2")

    c4, c5, c6 = st.columns(3)
    verdicts = [v for v in VERDICT_ORDER if v in set(df["verdict"])]
    sel_verdicts = c4.multiselect("Evidence verdict", verdicts, default=verdicts,
                                  key=f"al_{ns}_verdicts") if verdicts else []
    lifes = [l for l in LIFECYCLES if l in set(df["lifecycle"])]
    sel_life = c5.multiselect("Lifecycle", lifes, default=lifes, key=f"al_{ns}_life") if lifes else []
    all_status = [s for s in store.STATUSES if s in set(df["status"])] or store.STATUSES
    sel_status = c6.multiselect("Review status", all_status, default=all_status, key=f"al_{ns}_status")

    c7, c8, c9, c10 = st.columns([2, 2, 2, 2])
    min_conf = c7.slider("Minimum confidence (%)", 0, 100, 0, key=f"al_{ns}_conf")
    hide_urban = c8.checkbox("Hide urban lights", value=True, key=f"al_{ns}_urban",
                             disabled="land_cover" not in context_ok)
    pa_only = c9.checkbox("Forest & protected areas only", value=True, key=f"al_{ns}_pa",
                          disabled="protected_area" not in context_ok)
    sort_by = c10.selectbox(
        "Sort by",
        ["Severity, then confidence", "Confidence (high → low)", "Cluster size (large → small)",
         "Peak radiance (high → low)", "Newest first"],
        key=f"al_{ns}_sort",
    )

    mask = (df["category"].isin(sel_cats) & df["severity"].isin(sel_sevs)
            & df["status"].isin(sel_status) & (df["confidence"] >= min_conf))
    if verdicts:
        mask &= df["verdict"].isin(sel_verdicts)
    if lifes:
        mask &= df["lifecycle"].isin(sel_life)
    if hide_urban and "land_cover" in context_ok:
        mask &= ~df["urban_light"]
    if pa_only and "protected_area" in context_ok:
        mask &= df["zone"].ne("outside")
    out = df[mask]

    if search.strip():
        q = search.strip().lower()
        hay = (out["alert_id"] + " " + out["region"] + " " + out["assignee"].fillna("") + " "
               + out["note"].fillna("") + " " + out["lat"].round(4).astype(str)
               + " " + out["lon"].round(4).astype(str)).str.lower()
        out = out[hay.str.contains(q, regex=False)]

    sorters = {
        "Confidence (high → low)": ("confidence", False),
        "Cluster size (large → small)": ("cluster_size", False),
        "Peak radiance (high → low)": ("delta_max", False),
        "Newest first": ("first_seen", False),
    }
    if sort_by in sorters:
        col, asc = sorters[sort_by]
        out = out.sort_values(col, ascending=asc)
    return out.reset_index(drop=True)


def _table(df: pd.DataFrame, ns: str) -> pd.DataFrame:
    """Selectable table. Returns the rows the user ticked."""
    view = pd.DataFrame({
        "Select": False,
        "ID": df["alert_id"],
        "Severity": df["severity"].str.upper(),
        "Type": df["category"].map(category_label),
        "Verdict": df["verdict"],
        "Confidence %": df["confidence"],
        "Lifecycle": df["lifecycle"],
        "Seen ×": df["times_seen"],
        "Status": df["status"],
        "Assignee": df["assignee"],
        "Cluster px": df["cluster_size"],
        "Lat": df["lat"].round(5),
        "Lon": df["lon"].round(5),
        "Region": df["region"],
        "Evidence": df["evidence_text"],
        "Map": df["maps_url"],
    })

    select_all = st.checkbox("Select all alerts in this list", key=f"al_{ns}_selall")
    view["Select"] = select_all

    edited = st.data_editor(
        view,
        hide_index=True,
        height=min(520, 38 + 35 * max(len(view), 1)),
        key=f"al_{ns}_editor_{int(select_all)}",
        disabled=[c for c in view.columns if c != "Select"],
        column_config={
            "Select": st.column_config.CheckboxColumn("✓", width="small"),
            "Confidence %": st.column_config.ProgressColumn(
                "Confidence %", min_value=0, max_value=100, format="%d%%"),
            "Map": st.column_config.LinkColumn("Map", display_text="Open ↗"),
        },
        **WIDE,
    )
    picked_ids = edited.loc[edited["Select"], "ID"]
    return df[df["alert_id"].isin(picked_ids)]


def _review_panel(selected: pd.DataFrame) -> None:
    """Record the outcome of a field check for the ticked alerts."""
    st.markdown('<div class="section-title">📝 Review</div>', unsafe_allow_html=True)
    if selected.empty:
        st.caption("Tick one or more alerts in the table to set their status, assign them or add a note. "
                   "Decisions are saved with the alert and survive new scans.")
        return

    first = selected.iloc[0]
    if len(selected) == 1:
        try:
            hist = store.status_history(first["alert_id"], limit=20)
        except Exception:
            hist = None
        if hist is not None and not hist.empty:
            with st.expander(f"🕓  Review history of {first['alert_id']} ({len(hist)} change(s))"):
                st.dataframe(hist.drop(columns="alert_id"), hide_index=True)
    with st.form("review_form", clear_on_submit=False):
        st.markdown(f"**{len(selected)} alert(s) selected**")
        c1, c2 = st.columns(2)
        status = c1.selectbox("Status", store.STATUSES,
                              index=store.STATUSES.index(first["status"]) if first["status"] in store.STATUSES else 0)
        assignee = c2.text_input("Assigned to", value=first["assignee"] if len(selected) == 1 else "")
        c3, c4 = st.columns(2)
        true_cat = c3.selectbox(
            "Confirmed type (optional — used as the training label)",
            [""] + MODEL_CLASSES, format_func=lambda c: "— unchanged —" if not c else category_label(c),
            index=([""] + MODEL_CLASSES).index(first["true_category"]) if first["true_category"] in MODEL_CLASSES else 0,
        )
        note = c4.text_area("Note", value=first["note"] if len(selected) == 1 else "", height=90,
                            placeholder="e.g. Visited 14 Oct: 3 huts, ranger post informed")
        submitted = st.form_submit_button("💾  Save review", **WIDE)

    if submitted:
        try:
            changes = [{"alert_id": i, "status": status, "assignee": assignee, "note": note,
                        "true_category": true_cat} for i in selected["alert_id"]]
            store.set_status_many(changes)
        except Exception as e:
            st.error(f"Could not save the review: {e}")
        else:
            st.toast(f"Saved review for {len(changes)} alert(s).")
            st.rerun()


def _locations(map_df: pd.DataFrame, ticked: bool) -> None:
    st.markdown('<div class="section-title">📍 Exact location</div>', unsafe_allow_html=True)
    st.caption("Showing the ticked alerts." if ticked else "Showing every alert in the list above "
               "(tick rows to narrow the map).")
    render_location_explorer(map_df.reset_index(drop=True), key="al_loc")


def _share_links(target: pd.DataFrame, subject: str, body: str) -> None:
    wa = "https://wa.me/?text=" + quote(body[:1800])
    st.link_button("Share via WhatsApp", wa, **WIDE)


def _export_and_share(selected: pd.DataFrame, filtered: pd.DataFrame, all_alerts: List[Dict]) -> None:
    st.markdown('<div class="section-title">📤 Download or share</div>', unsafe_allow_html=True)

    scope_opts = [f"All alerts in the list above ({len(filtered)})"]
    if len(selected):
        scope_opts.insert(0, f"Only the ticked alerts ({len(selected)})")
    scope = st.radio("What to include", scope_opts, horizontal=True, key="al_scope")
    target = selected if scope.startswith("Only") else filtered

    if target.empty:
        st.info("Nothing to export — adjust the filters above.")
        return

    files = {
        "CSV (spreadsheet)": to_csv_bytes(target),
        "JSON (full detail)": to_json_bytes(target, all_alerts),
        "HTML report (print to PDF)": build_html_report(target),
    }

    tab_dl, tab_mail, tab_chat, tab_train = st.tabs(
        ["⬇️  Download", "✉️  E-mail", "💬  Telegram / WhatsApp", "🧠  Training labels"])

    with tab_dl:
        cols = st.columns(3)
        for col, (label, data) in zip(cols, files.items()):
            name, mime = FORMATS[label]
            col.download_button(f"⬇ {label}", data=data, file_name=name, mime=mime,
                                key=f"dl_{name}", **WIDE)
        st.caption("Tip: open the HTML report in your browser and choose Print → Save as PDF "
                   "for a shareable PDF.")
        with st.expander("Copy as plain text"):
            st.code(build_summary_text(target), language="text")

    with tab_mail:
        smtp = get_smtp_config(getattr(st, "secrets", None))
        to_raw = st.text_input("Send to (comma-separated)", key="mail_to",
                               placeholder="ranger@forestdept.gov.in, team@example.org")
        recipients, invalid = parse_recipients(to_raw)
        if invalid:
            st.warning("These don't look like e-mail addresses: " + ", ".join(invalid))

        subject = st.text_input(
            "Subject", key="mail_subject",
            value=f"TerraVision alert report — {len(target)} alert(s)",
        )
        note = st.text_area("Personal note (optional)", key="mail_note", height=80,
                            placeholder="Please verify the critical clusters this week…")
        body = (note.strip() + "\n\n" if note.strip() else "") + build_summary_text(target)
        attach = st.multiselect("Attach", list(FORMATS),
                                default=["CSV (spreadsheet)", "HTML report (print to PDF)"], key="mail_attach")

        with st.expander("Preview e-mail body"):
            st.code(body, language="text")

        send_col, link_col = st.columns([1, 2])
        with send_col:
            send_clicked = st.button("📨  Send now", key="mail_send", disabled=smtp is None, **WIDE)
        if smtp is None:
            st.info(
                "Direct sending isn't configured yet. Add `SMTP_HOST`, `SMTP_USER` and `SMTP_PASSWORD` "
                "to your `.env` (or Streamlit secrets) to enable **Send now** — see `.env.example`. "
                "Until then, use the buttons beside it to open your own mail app."
            )

        if send_clicked:
            if not recipients:
                st.error("Enter at least one valid recipient.")
            elif invalid:
                st.error("Fix the invalid addresses first.")
            else:
                attachments = {FORMATS[a][0]: (files[a], FORMATS[a][1]) for a in attach}
                try:
                    with st.spinner("Sending…"):
                        send_alert_email(smtp, recipients, subject, body, attachments)
                    st.success(f"Sent to {', '.join(recipients)} with {len(attachments)} attachment(s).")
                except Exception as e:
                    st.error(f"Could not send the e-mail: {e}")

        with link_col:
            if recipients and not invalid:
                m, g = st.columns(2)
                m.link_button("Open in mail app", mailto_link(recipients, subject, body), **WIDE)
                g.link_button("Compose in Gmail", gmail_link(recipients, subject, body), **WIDE)
            else:
                st.caption("Enter a valid recipient to enable the mail-app and Gmail buttons.")
        st.caption("Mail-app and Gmail links carry the summary text only (browsers can't attach files).")

    with tab_chat:
        text = build_summary_text(target, top_n=8)
        tg = get_telegram_config(getattr(st, "secrets", None))
        st.markdown("**Telegram** — posts the summary to your field team's chat.")
        if tg is None:
            st.info("Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env` (or Streamlit secrets) to "
                    "enable this. Create a bot with @BotFather, add it to your group, and see `.env.example`.")
        if st.button("📨  Send to Telegram", key="tg_send", disabled=tg is None, **WIDE):
            try:
                with st.spinner("Sending…"):
                    send_telegram(tg, text)
                st.success("Sent to Telegram.")
            except Exception as e:
                st.error(f"Could not send to Telegram: {e}")
        st.markdown("**WhatsApp** — opens WhatsApp with the summary ready to send to any chat.")
        _share_links(target, "", text)

    with tab_train:
        csv_bytes, n_rows, n_skipped = build_training_csv(all_alerts)
        st.markdown(
            "Alerts you mark **verified** (or **false positive** with a confirmed type) become labelled "
            "examples. Retrain the classifier on real, field-checked data with  \n"
            "`python model.py --csv terravision_labels.csv`")
        c1, c2 = st.columns(2)
        c1.metric("Labelled examples ready", n_rows)
        c2.metric("Reviewed but unusable", n_skipped,
                  help="Reviewed alerts from older scans without the full feature set, or without a usable type.")
        st.download_button("⬇ terravision_labels.csv", data=csv_bytes, file_name="terravision_labels.csv",
                           mime="text/csv", key="dl_labels", disabled=n_rows == 0, **WIDE)
        if n_rows == 0:
            st.caption("No labelled examples yet — set a status of “verified” on alerts you have checked.")
        elif n_rows < 100:
            st.caption("Tip: a few dozen examples per class is a start; hundreds make a real difference.")


def render_alerts_page(src: Dict) -> None:
    """`src` = {label, alerts, resolved, notes, context_ok} as built by app._alert_source()."""
    alerts: List[Dict] = src.get("alerts", [])
    st.markdown('<div class="section-title">🔔 Alerts</div>', unsafe_allow_html=True)
    st.caption(f"Data source: {src.get('label', '')}")

    if not alerts:
        st.info("No scan results yet. Open **Monitor** and run a satellite scan — "
                "real alerts appear here automatically and are saved for next time.")
        return

    for note in dict.fromkeys(src.get("notes", [])):
        st.info(note)

    full = to_dataframe(alerts)
    ns = str(abs(hash((src.get("label"), len(alerts)))) % 10**8)
    context_ok = src.get("context_ok", [])

    _severity_strip(full)
    filtered = _filters(full, ns, context_ok)
    st.markdown(f"**{len(filtered)}** of {len(full)} alerts shown")

    resolved = src.get("resolved", [])
    if resolved:
        with st.expander(f"✔️  {len(resolved)} alert(s) from the previous scan are no longer glowing"):
            st.dataframe(pd.DataFrame(resolved), hide_index=True)

    if filtered.empty:
        st.warning("No alerts match these filters.")
        return

    selected = _table(filtered, ns)
    _review_panel(selected)
    _locations(selected if len(selected) else filtered, ticked=bool(len(selected)))
    _export_and_share(selected, filtered, alerts)
