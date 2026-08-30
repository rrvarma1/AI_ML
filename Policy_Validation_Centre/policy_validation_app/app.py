from __future__ import annotations

import hashlib
import html
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from orchestrator import DealReviewOrchestrator


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "validation_history.db"
load_dotenv(APP_DIR / ".env")

st.set_page_config(page_title="Policy Validation Center", page_icon="✓", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: linear-gradient(145deg,#f8fbff 0%,#ffffff 48%,#f4fbf8 100%); color:#17223b; }
        .block-container { max-width: 1280px; padding-top:.55rem; padding-bottom:1rem; }
        [data-testid="stHeader"] { background: transparent; }
        .brand { display:flex; align-items:center; gap:11px; margin-bottom:.55rem; }
        .brand-mark { width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
            color:white; font-weight:800; font-size:18px; background:linear-gradient(135deg,#2563eb,#14b8a6);
            box-shadow:0 10px 28px rgba(37,99,235,.22); }
        .brand-name { font-size:17px; font-weight:700; color:#12203a; }
        .brand-sub { color:#718096; font-size:10px; letter-spacing:.04em; }
        .hero { min-height:78px; display:grid; grid-template-columns:240px minmax(0,1fr) 190px; align-items:center;
            overflow:hidden; border:1px solid #e6edf6; border-radius:16px; background:rgba(255,255,255,.95);
            box-shadow:0 8px 24px rgba(35,54,86,.05); margin-bottom:.45rem; }
        .hero-brand { display:flex; align-items:center; gap:10px; padding:10px 16px; border-right:1px solid #dbe4ef; }
        .hero-brand-mark { width:40px; height:40px; flex:0 0 40px; display:grid; place-items:center; border-radius:12px;
            color:white; font-weight:800; background:linear-gradient(135deg,#2563eb,#14b8a6); }
        .hero-brand-name { color:#12203a; font-size:16px; font-weight:700; }
        .hero-brand-sub { color:#718096; font-size:9px; margin-top:2px; }
        .hero-copy { padding:.65rem 1rem; z-index:2; }
        .hero-art { position:relative; min-height:78px; display:grid; place-items:center; overflow:hidden;
            background:linear-gradient(110deg,#fff,#eefaff); }
        .agent-art-grid { display:grid; grid-template-columns:repeat(4,30px); gap:7px; position:relative; }
        .agent-art-grid:before, .agent-art-grid:after { content:""; position:absolute; background:#b9d9ef; z-index:0; }
        .agent-art-grid:before, .agent-art-grid:after { display:none; }
        .agent-art-node { position:relative; z-index:1; width:30px; height:30px; display:flex;
            align-items:center; justify-content:center; border-radius:9px; background:#fff; color:#2563eb;
            border:1px solid #dbe8f5; box-shadow:0 10px 24px rgba(62,107,153,.14); }
        .agent-art-node:nth-child(2), .agent-art-node:nth-child(3) { color:#0f8b86; }
        .agent-art-node svg { width:15px; height:15px; fill:none; stroke:currentColor; stroke-width:1.8;
            stroke-linecap:round; stroke-linejoin:round; }
        .eyebrow { color:#2563eb; font-size:10px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
        .hero h1 { font-size:20px; line-height:1.1; letter-spacing:-.025em; margin:.18rem 0 .15rem; color:#12203a; }
        .hero p { color:#2563eb; max-width:760px; margin:0; font-size:10px; }
        .new-validation-panel-title { color:#17223b; font-size:14px; font-weight:700; margin-bottom:2px; }
        .new-validation-panel-note { color:#718096; font-size:10px; margin-bottom:7px; }
        .final-evaluation-status { display:flex; align-items:center; gap:7px; padding:8px 10px; margin-bottom:8px;
            border-radius:9px; background:#ecfdf3; color:#15803d; font-size:12px; font-weight:700; }
        .final-evaluation-status.pending { background:#f1f5f9; color:#64748b; }
        .final-evaluation-status.in-progress { background:#eff6ff; color:#1d4ed8; }
        .final-evaluation-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px 10px; }
        .final-evaluation-label { color:#718096; font-size:9px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
        .final-evaluation-value { color:#17223b; font-size:12px; font-weight:700; line-height:1.25; margin-top:2px; overflow-wrap:anywhere; }
        .st-key-deal_officer_panel { border-left:1px solid #dbe4ef; padding-left:10px; height:100%; }
        .officer-status { margin-top:7px; padding:7px 9px; border-radius:9px; background:#f1f5f9; color:#64748b; font-size:11px; font-weight:700; }
        .officer-status.active { background:#fff7ed; color:#b45309; }
        .officer-status.approve { background:#dcfce7; color:#15803d; }
        .officer-status.reject { background:#fee2e2; color:#b91c1c; }
        .officer-status.on-hold { background:#fef3c7; color:#b45309; }
        .st-key-deal_officer_panel div.stButton > button {
            min-height:32px !important; height:32px !important; padding:.25rem .3rem !important;
            border-radius:8px !important; white-space:nowrap !important; word-break:keep-all !important;
        }
        .st-key-deal_officer_panel div.stButton > button p {
            margin:0 !important; color:inherit !important; font-size:12px !important; line-height:1 !important;
            font-weight:700 !important; white-space:nowrap !important; word-break:keep-all !important;
        }
        .deal-condition-list { list-style:none; padding:0; margin:2px 0 0; display:grid; grid-template-columns:1fr 1fr; gap:6px 12px; }
        .deal-condition-list li { position:relative; padding-left:16px; color:#52647b; font-size:11.5px; line-height:1.38; }
        .deal-condition-list li:before { content:"✓"; position:absolute; left:0; top:0; color:#15803d; font-weight:800; }
        .deal-condition-list li:first-child { grid-column:1 / -1; }
        [data-testid="stFileUploader"] { background:#fbfdff; border:1.5px dashed #b8cae6; border-radius:14px; padding:7px 8px; }
        [data-testid="stFileUploaderDropzone"] { min-height:44px; padding:.45rem .65rem; }
        [data-testid="stFileUploaderDropzoneInstructions"] { display:none; }
        [data-testid="stFileUploaderDropzone"] button { margin:0; }
        [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) { padding:6px 8px; }
        [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) [data-testid="stFileUploaderDropzone"] { display:none; }
        [data-testid="stFileUploaderFile"] { min-height:42px; padding:5px 7px; }
        .st-key-new_validation_upload [data-testid="stVerticalBlock"] { gap:.45rem; }
        .st-key-new_validation_upload [data-testid="stFileUploader"] { margin:0; }
        .st-key-new_validation_upload div.stButton > button { width:calc(100% - 16px) !important;
            margin-left:8px; margin-right:8px; padding:.55rem .8rem; }
        div.stButton > button:disabled,
        div.stButton > button[kind="primary"]:disabled {
            color:#526174 !important; background:#e8eef6 !important; border:1px solid #c9d5e5 !important;
            opacity:1 !important; box-shadow:none !important; cursor:not-allowed;
        }
        div.stButton > button:disabled p,
        div.stButton > button[kind="primary"]:disabled p {
            color:#526174 !important; opacity:1 !important; font-weight:700 !important;
        }
        [data-testid="stMetric"] { background:white; border:1px solid #e7edf5; border-radius:16px; padding:16px 18px;
            box-shadow:0 8px 25px rgba(30,50,80,.045); }
        .risk-summary-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:.4rem 0 .7rem; }
        .risk-summary-box { min-width:0; min-height:92px; display:flex; flex-direction:column; justify-content:center;
            padding:14px 16px; background:#fff; border:1px solid #e7edf5; border-radius:15px;
            box-shadow:0 7px 22px rgba(30,50,80,.045); }
        .risk-summary-label { color:#64748b; font-size:12px; font-weight:600; margin-bottom:7px; }
        .risk-summary-value { color:#17223b; font-size:19px; line-height:1.2; font-weight:700;
            overflow-wrap:anywhere; }
        .execution-overview-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:11px; margin:.45rem 0 .7rem; }
        .execution-card { min-width:0; padding:14px; border:1px solid #e4ebf4; border-radius:15px; background:#fff;
            box-shadow:0 7px 22px rgba(30,50,80,.045); }
        .execution-card-head { display:flex; align-items:center; gap:9px; margin-bottom:11px; }
        .execution-card-icon { width:32px; height:32px; flex:0 0 32px; display:grid; place-items:center; border-radius:10px;
            background:#edf2f7; color:#526174; font-size:10px; font-weight:800; }
        .execution-card-name { color:#17223b; font-size:13px; font-weight:700; }
        .execution-status-pill { display:inline-flex; align-items:center; gap:5px; width:max-content; margin-top:3px;
            padding:3px 7px; border-radius:999px; font-size:10px; font-weight:700; }
        .execution-status-pill.success { color:#15803d; background:#dcfce7; }
        .execution-status-pill.failure { color:#b91c1c; background:#fee2e2; }
        .execution-status-pill.in_progress { color:#1d4ed8; background:#dbeafe; }
        .execution-status-pill.not_run { color:#64748b; background:#f1f5f9; }
        .execution-card-summary { color:#64748b; font-size:11px; line-height:1.45; min-height:32px; }
        .results-dashboard { margin-top:.4rem; }
        .latest-strip { display:flex; align-items:center; gap:9px; min-height:36px; padding:6px 10px; border:1px solid #d9eee4;
            border-radius:11px; background:#f3fbf7; }
        .latest-strip-check { color:#15803d; font-size:14px; font-weight:800; }
        .latest-strip-title { color:#166534; font-size:11px; font-weight:700; }
        .latest-strip-meta { color:#64748b; font-size:9px; margin-left:5px; }
        .latest-side { display:flex; flex-direction:column; gap:5px; justify-content:center; min-height:88px; }
        .latest-side-status { color:#15803d; font-size:12px; font-weight:700; }
        .latest-side-id { color:#17223b; font-size:14px; font-weight:700; }
        .latest-side-file { color:#64748b; font-size:10px; line-height:1.35; overflow-wrap:anywhere; }
        .validation-summary-head { display:flex; align-items:center; gap:7px; color:#15803d;
            font-size:14px; font-weight:700; margin-bottom:10px; }
        .validation-summary-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px 12px; margin-bottom:9px; }
        .validation-summary-label { color:#718096; font-size:9px; font-weight:600; text-transform:uppercase;
            letter-spacing:.05em; }
        .validation-summary-value { color:#17223b; font-size:11px; font-weight:600; line-height:1.3;
            margin-top:2px; overflow-wrap:anywhere; }
        .validation-summary-value.success { color:#15803d; }
        .dashboard-panel-title { color:#17223b; font-size:17px; font-weight:700; margin-bottom:10px; }
        .dashboard-panel-note { color:#718096; font-size:11px; float:right; font-weight:500; }
        .overview-quad { display:grid; grid-template-columns:1fr 1fr; }
        .overview-item { display:flex; align-items:center; gap:9px; min-width:0; padding:9px 11px; }
        .overview-item:nth-child(odd) { border-right:1px solid #dbe4ef; }
        .overview-item:nth-child(-n+2) { border-bottom:1px solid #dbe4ef; }
        .overview-icon { width:32px; height:32px; flex:0 0 32px; display:grid; place-items:center; border-radius:9px;
            background:#edf2f7; color:#526174; }
        .overview-icon svg { width:17px; height:17px; fill:none; stroke:currentColor; stroke-width:1.8;
            stroke-linecap:round; stroke-linejoin:round; }
        .overview-name { color:#17223b; font-size:13px; font-weight:700; }
        .overview-status { font-size:11px; font-weight:600; margin-top:2px; line-height:1.25; }
        .overview-status.success { color:#15803d; } .overview-status.failure { color:#b91c1c; }
        .overview-status.in_progress { color:#1d4ed8; } .overview-status.not_run { color:#64748b; }
        .overview-status.non_compliant { color:#b91c1c; }
        .overview-status.insufficient_data { color:#b45309; }
        .overview-status.compliant, .overview-status.acceptable_risk { color:#15803d; }
        .overview-status.adverse_risk { color:#b91c1c; }
        .execution-detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:11px 14px; }
        .execution-detail-item { border-left:2px solid #dbe4ef; padding-left:8px; min-width:0; }
        .execution-detail-name { color:#17223b; font-size:13px; font-weight:700; }
        .execution-detail-copy { color:#64748b; font-size:11px; line-height:1.4; overflow-wrap:anywhere; }
        .result-section-heading { color:#18243b; font-size:17px; font-weight:700; margin:.75rem 0 .05rem; }
        .result-section-note { color:#718096; font-size:11px; margin-bottom:.4rem; }
        .review-outcome-grid { display:grid; grid-template-rows:auto auto; margin:.05rem 0; }
        .review-outcome-card { min-width:0; padding:4px 10px 7px; }
        .review-outcome-card:first-child { border-bottom:1px solid #dbe4ef; }
        .review-outcome-title { color:#17223b; font-size:13px; font-weight:700; margin-bottom:5px; }
        .review-stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
        .review-stat-label { color:#718096; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
        .review-stat-value { color:#17223b; font-size:16px; font-weight:700; line-height:1.25; overflow-wrap:anywhere; }
        .risk-outcome-row { display:grid; grid-template-columns:.8fr 1.7fr .8fr; gap:9px; }
        .review-summary-copy { color:#52647b; font-size:12px; line-height:1.35; margin-top:5px; }
        .detail-selector-note { color:#64748b; font-size:11px; margin:.2rem 0 .15rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; border-color:#e4ebf4;
            box-shadow:0 5px 16px rgba(30,50,80,.035); }
        [data-testid="stSegmentedControl"] { margin-top:.15rem; }
        [data-testid="stSegmentedControl"] button { font-size:11px; min-height:36px; padding:.3rem .6rem; }
        [class*="st-key-supporting_details_"] div.stButton > button { min-height:48px; width:100%; justify-content:flex-start;
            padding:10px 13px; border:1px solid #dbeafe; border-radius:11px; background:#eaf4ff; color:#2583ee;
            font-size:11px; font-weight:500; box-shadow:none; }
        [class*="st-key-supporting_details_"] div.stButton > button:hover { border-color:#93c5fd; background:#dceeff; color:#1769d2; }
        [class*="st-key-supporting_details_"] div.stButton > button svg { color:currentColor; }
        .risk-detail-pane { margin-top:10px; padding:13px; border:1px solid #e4ebf4; border-radius:14px;
            background:#fff; box-shadow:0 5px 16px rgba(30,50,80,.035); }
        .risk-detail-heading { color:#17223b; font-size:17px; font-weight:700; margin-bottom:9px; }
        .risk-detail-heading span { float:right; color:#718096; font-size:11px; font-weight:500; }
        .risk-detail-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
        .risk-detail-card { min-width:0; height:142px; box-sizing:border-box; padding:11px 12px; border-radius:11px;
            background:#f8fafc; border:1px solid #edf1f6; }
        .risk-detail-title { color:#17223b; font-size:13px; font-weight:700; margin-bottom:7px; }
        .risk-detail-status { color:#475569; font-size:12px; font-weight:600; margin-bottom:5px; }
        .risk-detail-copy { color:#64748b; font-size:11px; line-height:1.4; }
        .risk-detail-scroll { max-height:98px; overflow-y:auto; padding-right:5px; scrollbar-width:thin; }
        .risk-detail-list { margin:0; padding-left:17px; color:#526174; font-size:11px; line-height:1.4; }
        .risk-detail-list li { margin-bottom:5px; }
        [data-testid="stExpander"] { margin:.25rem 0 .35rem; }
        [data-testid="stExpander"] details summary { padding:.45rem .7rem; }
        [data-testid="stExpanderDetails"] { padding-top:.2rem; padding-bottom:.35rem; }
        .section-title { font-size:17px; font-weight:700; color:#18243b; margin:.45rem 0 .05rem; }
        .section-note { color:#718096; font-size:11px; margin-bottom:.35rem; }
        .execution-status { min-height:16px; margin:.05rem 0 .08rem; color:#64748b; font-size:11px; line-height:1.25; }
        .execution-status.in_progress { color:#1d4ed8; } .execution-status.completed { color:#15803d; }
        .execution-status.failed { color:#b91c1c; }
        .workflow-heading { color:#18243b; font-size:16px; font-weight:700; margin-top:-.1rem; margin-bottom:0; }
        .workflow-note { color:#718096; font-size:11px; margin-bottom:3px; line-height:1.2; }
        .workflow-graph { --workflow-copy-size:11px; display:grid; grid-template-columns:minmax(190px,1fr) 24px minmax(190px,1fr) 50px minmax(270px,1.25fr);
            align-items:center; gap:8px; padding:3px 0 0; }
        .workflow-graph.has-officer { grid-template-columns:minmax(155px,.9fr) 20px minmax(155px,.9fr) 42px minmax(235px,1.2fr) 20px minmax(165px,.95fr); }
        .workflow-step { flex:1 1 0; min-width:0; display:flex; }
        .workflow-node { width:30px; height:30px; flex:0 0 30px; display:grid; place-items:center; border-radius:9px;
            color:#64748b; background:#e9eef5; font-weight:800; font-size:9px; box-shadow:0 0 0 1px #dbe4ef; }
        .workflow-panel { flex:1; min-width:0; min-height:78px; display:flex; flex-direction:column;
            padding:6px 8px; border-radius:12px; border:1px solid #e3eaf3; background:#fff;
            box-shadow:0 5px 15px rgba(30,50,80,.035); transition:all .2s ease; }
        .workflow-top { display:flex; align-items:center; gap:8px; }
        .workflow-step.in_progress .workflow-panel { border-color:#93c5fd; background:#eff6ff; box-shadow:0 7px 20px rgba(37,99,235,.10); }
        .workflow-step.completed .workflow-panel { border-color:#86efac; background:#f0fdf4; box-shadow:0 7px 20px rgba(22,163,74,.08); }
        .workflow-step.failed .workflow-panel { border-color:#fca5a5; background:#fef2f2; box-shadow:0 7px 20px rgba(220,38,38,.08); }
        .workflow-step.interrupted .workflow-panel { border-color:#fcd34d; background:#fffbeb; box-shadow:0 7px 20px rgba(217,119,6,.08); }
        .workflow-step.non_compliant .workflow-panel { border-color:#fca5a5; background:#fef2f2; box-shadow:0 7px 20px rgba(220,38,38,.08); }
        .workflow-step.insufficient_data .workflow-panel { border-color:#fcd34d; background:#fffbeb; box-shadow:0 7px 20px rgba(217,119,6,.08); }
        .workflow-step.compliant .workflow-panel, .workflow-step.acceptable_risk .workflow-panel { border-color:#86efac; background:#f0fdf4; box-shadow:0 7px 20px rgba(22,163,74,.08); }
        .workflow-step.adverse_risk .workflow-panel { border-color:#fca5a5; background:#fef2f2; box-shadow:0 7px 20px rgba(220,38,38,.08); }
        .workflow-step.in_progress .workflow-node { color:#1d4ed8; background:#dbeafe; box-shadow:0 0 0 1px #60a5fa; }
        .workflow-step.completed .workflow-node { color:#15803d; background:#dcfce7; box-shadow:0 0 0 1px #4ade80; }
        .workflow-step.failed .workflow-node { color:#b91c1c; background:#fee2e2; box-shadow:0 0 0 1px #f87171; }
        .workflow-step.interrupted .workflow-node { color:#b45309; background:#fef3c7; box-shadow:0 0 0 1px #fbbf24; }
        .workflow-step.non_compliant .workflow-node { color:#b91c1c; background:#fee2e2; box-shadow:0 0 0 1px #f87171; }
        .workflow-step.insufficient_data .workflow-node { color:#b45309; background:#fef3c7; box-shadow:0 0 0 1px #fbbf24; }
        .workflow-step.compliant .workflow-node, .workflow-step.acceptable_risk .workflow-node { color:#15803d; background:#dcfce7; box-shadow:0 0 0 1px #4ade80; }
        .workflow-step.adverse_risk .workflow-node { color:#b91c1c; background:#fee2e2; box-shadow:0 0 0 1px #f87171; }
        .workflow-step.officer_pending .workflow-panel, .workflow-step.officer_hold .workflow-panel { border-color:#fcd34d; background:#fffbeb; }
        .workflow-step.officer_approve .workflow-panel { border-color:#86efac; background:#f0fdf4; }
        .workflow-step.officer_reject .workflow-panel { border-color:#fca5a5; background:#fef2f2; }
        .workflow-step.officer_pending .workflow-node, .workflow-step.officer_hold .workflow-node { color:#b45309; background:#fef3c7; }
        .workflow-step.officer_approve .workflow-node { color:#15803d; background:#dcfce7; }
        .workflow-step.officer_reject .workflow-node { color:#b91c1c; background:#fee2e2; }
        .workflow-connector { width:24px; height:3px; position:relative;
            background:#dbe4ef; border-radius:99px; }
        .workflow-connector:after { content:""; position:absolute; right:-1px; top:-4px; border-left:7px solid #dbe4ef;
            border-top:5px solid transparent; border-bottom:5px solid transparent; }
        .workflow-connector.in_progress { background:#60a5fa; } .workflow-connector.in_progress:after { border-left-color:#60a5fa; }
        .workflow-connector.completed { background:#4ade80; } .workflow-connector.completed:after { border-left-color:#4ade80; }
        .workflow-connector.failed { background:#f87171; } .workflow-connector.failed:after { border-left-color:#f87171; }
        .workflow-branch { align-self:stretch; min-height:162px; position:relative; color:#dbe4ef; }
        .workflow-branch span { position:absolute; display:block; background:currentColor; border-radius:99px; }
        .workflow-branch .branch-in { left:0; top:50%; width:50%; height:3px; }
        .workflow-branch .branch-rail { left:50%; top:25%; width:3px; height:50%; }
        .workflow-branch .branch-top { left:50%; top:25%; width:50%; height:3px; }
        .workflow-branch .branch-bottom { left:50%; top:75%; width:50%; height:3px; }
        .workflow-branch .branch-top:after, .workflow-branch .branch-bottom:after { content:""; position:absolute;
            right:-1px; top:-4px; border-left:7px solid currentColor; border-top:5px solid transparent;
            border-bottom:5px solid transparent; }
        .workflow-branch.in_progress { color:#60a5fa; } .workflow-branch.completed { color:#4ade80; }
        .workflow-branch.failed { color:#f87171; }
        .workflow-branch.non_compliant { color:#f87171; }
        .workflow-branch.insufficient_data { color:#fbbf24; }
        .workflow-branch.compliant, .workflow-branch.acceptable_risk { color:#4ade80; }
        .workflow-branch.adverse_risk { color:#f87171; }
        .workflow-parallel { display:grid; grid-template-rows:1fr 1fr; gap:5px; }
        .workflow-parallel-label { color:#64748b; font-size:9px; font-weight:700; letter-spacing:.07em;
            text-align:center; text-transform:uppercase; margin-bottom:2px; }
        .workflow-parallel .workflow-panel { min-height:76px; }
        .workflow-copy { flex:1; min-width:0; }
        .workflow-name { color:#17223b; font-size:11px !important; font-weight:700; line-height:1.2; }
        .workflow-status { color:#64748b; font-size:9.5px; margin-top:1px; }
        .workflow-panel .workflow-activities { margin:3px 0 0; padding:3px 0 0 15px; border-top:1px solid rgba(148,163,184,.25);
            color:#526174; font-size:11px !important; font-weight:500; line-height:1.15 !important; }
        .workflow-panel .workflow-activities > li { margin:0; padding:0; font-size:11px !important; line-height:1.15 !important; }
        .workflow-panel .workflow-activities > li::marker { font-size:11px !important; }
        .in_progress .workflow-status { color:#1d4ed8; } .completed .workflow-status { color:#15803d; }
        .failed .workflow-status { color:#b91c1c; }
        .workflow-spinner { width:15px; height:15px; flex:0 0 15px; border:2px solid #bfdbfe;
            border-top-color:#2563eb; border-radius:50%; animation:workflow-spin .8s linear infinite; }
        .workflow-check, .workflow-fail { width:18px; height:18px; flex:0 0 18px; display:grid; place-items:center;
            border-radius:50%; color:white; font-size:10px; font-weight:800; }
        .workflow-check { background:#16a34a; } .workflow-fail { background:#dc2626; }
        @keyframes workflow-spin { to { transform:rotate(360deg); } }
        .result-card { border-radius:18px; padding:19px 20px; background:white; border:1px solid #e7edf5;
            box-shadow:0 8px 26px rgba(35,54,86,.05); margin:.55rem 0; }
        .result-card.pass { border-left:5px solid #10b981; }
        .result-card.warn { border-left:5px solid #f59e0b; }
        .result-card.fail { border-left:5px solid #ef4444; }
        .card-title { font-weight:700; color:#1e293b; margin-bottom:5px; }
        .card-copy { color:#64748b; font-size:14px; }
        .pill { display:inline-block; padding:4px 9px; border-radius:999px; font-size:11px; font-weight:700; margin-right:8px; }
        .pill.pass { background:#dcfce7; color:#047857; } .pill.warn { background:#fef3c7; color:#b45309; }
        .pill.fail { background:#fee2e2; color:#b91c1c; }
        .history-review-head { display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:8px; margin:.45rem 0 .65rem; }
        .history-review-stat { min-width:0; padding:9px 10px; border:1px solid #e4ebf4; border-radius:11px; background:#f8fafc; }
        .history-review-label { color:#718096; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
        .history-review-value { color:#17223b; font-size:12px; font-weight:700; line-height:1.3; margin-top:3px; overflow-wrap:anywhere; }
        .history-term-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }
        .history-term { padding:9px 10px; border:1px solid #e7edf5; border-radius:10px; background:#fbfdff; }
        .history-term-label { color:#64748b; font-size:10px; font-weight:600; }
        .history-term-value { color:#17223b; font-size:13px; font-weight:700; margin-top:3px; overflow-wrap:anywhere; }
        .history-finding { padding:10px 12px; margin-bottom:7px; border:1px solid #e7edf5; border-left:4px solid #94a3b8; border-radius:10px; background:white; }
        .history-finding.pass { border-left-color:#22c55e; } .history-finding.fail { border-left-color:#ef4444; }
        .history-finding.warn { border-left-color:#f59e0b; }
        .history-finding-title { color:#17223b; font-size:12px; font-weight:700; }
        .history-finding-copy { color:#52647b; font-size:11px; line-height:1.45; margin-top:4px; }
        @media (max-width:900px) { .history-review-head { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .history-term-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        div.stButton > button[kind="primary"] { border:0; border-radius:12px; background:linear-gradient(135deg,#2563eb,#0d9488);
            font-weight:700; padding:.68rem 1.25rem; box-shadow:0 8px 20px rgba(37,99,235,.2); }
        [data-testid="stDownloadButton"] > button { border:0; border-radius:12px;
            background:linear-gradient(135deg,#2563eb,#0d9488); color:white; font-weight:700;
            padding:.68rem 1.25rem; box-shadow:0 8px 20px rgba(37,99,235,.2); }
        [data-testid="stDownloadButton"] > button p { color:white; }
        [data-testid="stDownloadButton"] > button:hover { border:0; color:white;
            background:linear-gradient(135deg,#1d4ed8,#0f766e); }
        div.stButton > button { border-radius:12px; }
        [data-testid="stTabs"] button { font-weight:600; }
        footer { visibility:hidden; }
        @media (max-width: 800px) { .hero { grid-template-columns:1fr; } .hero-art { display:none; }
            .risk-summary-grid { grid-template-columns:1fr; }
            .execution-overview-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .review-outcome-grid { grid-template-columns:1fr; }
            .risk-detail-grid { grid-template-columns:1fr; }
            .risk-detail-card { height:auto; min-height:142px; }
            .workflow-graph { display:flex; flex-direction:column; align-items:stretch; }
            .workflow-connector { width:3px; height:22px; margin-left:19px; }
            .workflow-connector:after { right:-4px; top:auto; bottom:-1px; border-left:5px solid transparent;
                border-right:5px solid transparent; border-top:7px solid #dbe4ef; border-bottom:0; }
            .workflow-connector.in_progress:after { border-left-color:transparent; border-top-color:#60a5fa; }
            .workflow-connector.completed:after { border-left-color:transparent; border-top-color:#4ade80; }
            .workflow-connector.failed:after { border-left-color:transparent; border-top-color:#f87171; }
            .workflow-branch { width:3px; min-height:22px; height:22px; margin-left:19px; background:currentColor; }
            .workflow-branch span { display:none; }
            .workflow-parallel { grid-template-rows:auto; }
        }
        @media (max-width: 480px) { .execution-overview-grid { grid-template-columns:1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                score INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                warnings INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                findings_json TEXT NOT NULL,
                terms_json TEXT NOT NULL DEFAULT '{}',
                policy_filename TEXT,
                agent_metadata_json TEXT NOT NULL DEFAULT '{}',
                risk_json TEXT NOT NULL DEFAULT '{}',
                officer_decision TEXT
            )
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(validations)")}
        if "terms_json" not in columns:
            conn.execute("ALTER TABLE validations ADD COLUMN terms_json TEXT NOT NULL DEFAULT '{}'")
        if "policy_filename" not in columns:
            conn.execute("ALTER TABLE validations ADD COLUMN policy_filename TEXT")
        if "agent_metadata_json" not in columns:
            conn.execute("ALTER TABLE validations ADD COLUMN agent_metadata_json TEXT NOT NULL DEFAULT '{}'")
        if "risk_json" not in columns:
            conn.execute("ALTER TABLE validations ADD COLUMN risk_json TEXT NOT NULL DEFAULT '{}'")
        if "officer_decision" not in columns:
            conn.execute("ALTER TABLE validations ADD COLUMN officer_decision TEXT")
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if schema_version < 2:
            conn.execute("""
                UPDATE validations
                SET officer_decision = CASE
                    WHEN status = 'COMPLIANT' THEN 'Approve'
                    WHEN status = 'NON_COMPLIANT' THEN 'Rejected'
                    ELSE officer_decision
                END
                WHERE officer_decision IS NULL OR trim(officer_decision) = ''
            """)
            conn.execute("PRAGMA user_version = 2")


def extract_text(uploaded_file) -> str:
    payload = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
    return payload.decode("utf-8", errors="replace")


WORKFLOW_STAGES = {
    "orchestrator": ("OR", "Orchestrator", ["Initializes the review case", "Coordinates stage handoffs", "Tracks pipeline execution"]),
    "extractor": ("EX", "Extractor", ["Reads the deal document", "Extracts structured terms", "Creates the JSON handoff"]),
    "compliance_validator": ("CV", "Compliance Validator", ["Loads versioned rules", "Evaluates policy thresholds", "Returns findings and score"]),
    "risk_summary": ("A2", "Risk & Summary", ["Calculates financial ratios", "Flags deterministic risks", "Generates a grounded summary"]),
}


AGENT_ICONS = {
    "orchestrator": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 7.5l4 8M17 7.5l-4 8M7 6h10"/></svg>',
    "extractor": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v6M6 3v18h7M14 3v5h5"/><circle cx="17" cy="17" r="3"/><path d="M19.3 19.3L22 22"/></svg>',
    "compliance_validator": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v18M5 7h14M7 7l-4 7h8L7 7zm10 0l-4 7h8l-4-7zM8 21h8"/></svg>',
    "risk_summary": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20V10M10 20V5M16 20v-8M22 20H2"/><path d="M3 7l6-4 6 5 6-5"/></svg>',
}


def hero_markup() -> str:
    return (
        '<div class="hero"><div class="hero-brand"><div class="hero-brand-mark">✓</div><div>'
        '<div class="hero-brand-name">Policy Validation Center</div><div class="hero-brand-sub">Intelligent deal validation</div>'
        '</div></div><div class="hero-copy"><div class="eyebrow">Multi-agent deal review</div>'
        '<h1>Move from documents to decisions.</h1><p>Our multi agent workflow extracts, validates, checks policy compliance, '
        'and summarizes findings—so you can decide with confidence.</p></div>'
        '<div class="hero-art" role="img" aria-label="Four-agent workflow illustration"><div class="agent-art-grid">'
        f'<div class="agent-art-node">{AGENT_ICONS["orchestrator"]}</div>'
        f'<div class="agent-art-node">{AGENT_ICONS["extractor"]}</div>'
        f'<div class="agent-art-node">{AGENT_ICONS["compliance_validator"]}</div>'
        f'<div class="agent-art-node">{AGENT_ICONS["risk_summary"]}</div>'
        '</div></div></div>'
    )


def final_evaluation_markup(result: dict | None = None, risk_result: dict | None = None, state: str = "pending") -> str:
    risk_result = risk_result or {}
    if state == "in_progress":
        status_text, status_class = "Evaluation in progress", "in-progress"
    elif result:
        status_text, status_class = "Review completed", ""
    else:
        status_text, status_class = "Awaiting validation", "pending"
    overall_risk = str(risk_result.get("overall_risk", "—")).replace("_", " ").title()
    recommendation = str(risk_result.get("recommendation", "—")).replace("_", " ").title()
    confidence = f'{100 * float(risk_result.get("confidence", 0)):.0f}%' if risk_result else "—"
    compliance = str(result.get("status", "—")).replace("_", " ").title() if result else "—"
    return (
        '<div class="new-validation-panel-title">System Validation</div>'
        '<div class="new-validation-panel-note">Policy Validation Center Review Comments</div>'
        f'<div class="final-evaluation-status {status_class}">● {html.escape(status_text)}</div>'
        '<div class="final-evaluation-grid">'
        f'<div><div class="final-evaluation-label">Overall risk</div><div class="final-evaluation-value">{html.escape(overall_risk)}</div></div>'
        f'<div><div class="final-evaluation-label">Recommendation</div><div class="final-evaluation-value">{html.escape(recommendation)}</div></div>'
        f'<div><div class="final-evaluation-label">Confidence</div><div class="final-evaluation-value">{confidence}</div></div>'
        f'<div><div class="final-evaluation-label">Compliance</div><div class="final-evaluation-value">{html.escape(compliance)}</div></div>'
        '</div>'
    )


def requires_deal_officer(result: dict | None, risk_result: dict | None) -> bool:
    if not result or not risk_result:
        return False
    compliance_adverse = result.get("status") != "COMPLIANT"
    risk_adverse = not (
        risk_result.get("overall_risk") == "LOW" and risk_result.get("recommendation") == "PROCEED"
    )
    return compliance_adverse or risk_adverse


def save_officer_decision(validation_id: str, decision: str | None, agent_metadata: dict | None = None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        if agent_metadata is None:
            conn.execute("UPDATE validations SET officer_decision = ? WHERE validation_id = ?", (decision, validation_id))
        else:
            conn.execute(
                "UPDATE validations SET officer_decision = ?, agent_metadata_json = ? WHERE validation_id = ?",
                (decision, json.dumps(agent_metadata), validation_id),
            )


def load_officer_decision(validation_id: str | None) -> str | None:
    if not validation_id:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT officer_decision FROM validations WHERE validation_id = ?", (validation_id,)
        ).fetchone()
    return row[0] if row else None


def deal_conditions_markup(result: dict | None = None, risk_result: dict | None = None) -> str:
    candidates: list[str] = []
    if result:
        for finding in result.get("findings", []):
            if finding.get("status") in {"FAIL", "INSUFFICIENT_DATA"}:
                candidates.append(str(finding.get("detail", finding.get("title", "Review compliance finding"))).split(" Actual:", 1)[0])
    for flag in (risk_result or {}).get("risk_flags", []):
        candidates.append(str(flag.get("recommended_action") or flag.get("title") or "Review risk finding"))
    for missing in (risk_result or {}).get("missing_information", []):
        candidates.append(f"Obtain missing information: {missing}")
    unique_conditions = list(dict.fromkeys(item.strip() for item in candidates if item.strip()))[:5]
    if not unique_conditions:
        unique_conditions = ["Critical deal conditions will appear after the review completes."]
    items = "".join(f"<li>{html.escape(item)}</li>" for item in unique_conditions)
    return (
        '<div class="new-validation-panel-title">Key deal conditions</div>'
        '<div class="new-validation-panel-note">Up to five critical conditions from the final review</div>'
        f'<ul class="deal-condition-list">{items}</ul>'
    )


def workflow_timeline(states: dict[str, str], show_deal_officer: bool = False, officer_decision: str | None = None) -> str:
    labels = {
        "idle": "Ready", "in_progress": "In progress", "completed": "Completed", "failed": "Failed", "interrupted": "Paused for decision",
        "non_compliant": "Non compliant", "insufficient_data": "Insufficient data",
        "compliant": "Compliant", "acceptable_risk": "Low risk", "adverse_risk": "Adverse risk",
    }

    def stage_card(stage: str) -> str:
        icon, name, activities = WORKFLOW_STAGES[stage]
        status = states[stage]
        indicator = (
            '<span class="workflow-spinner"></span>' if status == "in_progress"
            else '<span class="workflow-check">✓</span>' if status in {"completed", "compliant", "acceptable_risk"}
            else '<span class="workflow-fail">!</span>' if status in {"failed", "non_compliant", "insufficient_data", "adverse_risk"}
            else ""
        )
        activity_items = "".join(f"<li>{activity}</li>" for activity in activities)
        return (
            f'<div class="workflow-step {status}"><div class="workflow-panel"><div class="workflow-top">'
            f'<div class="workflow-node">{icon}</div><div class="workflow-copy"><div class="workflow-name">{name}</div>'
            f'<div class="workflow-status">{labels[status]}</div></div>{indicator}</div>'
            f'<ul class="workflow-activities">{activity_items}</ul></div></div>'
        )

    parallel_states = [states["compliance_validator"], states["risk_summary"]]
    branch_status = (
        "failed" if "failed" in parallel_states
        else "adverse_risk" if "adverse_risk" in parallel_states
        else "non_compliant" if "non_compliant" in parallel_states
        else "insufficient_data" if "insufficient_data" in parallel_states
        else "in_progress" if "in_progress" in parallel_states
        else "completed" if all(status in {"completed", "compliant", "acceptable_risk"} for status in parallel_states)
        else "idle"
    )
    officer_state = {
        "Approve": ("officer_approve", "Approved"),
        "Reject": ("officer_reject", "Rejected"),
        "On Hold": ("officer_hold", "On hold"),
    }.get(officer_decision, ("officer_pending", "Decision required"))
    officer_card = (
        f'<div class="workflow-connector {branch_status}"></div>'
        f'<div class="workflow-step {officer_state[0]}"><div class="workflow-panel"><div class="workflow-top">'
        '<div class="workflow-node">DO</div><div class="workflow-copy"><div class="workflow-name">Deal Officer Validator</div>'
        f'<div class="workflow-status">{officer_state[1]}</div></div></div>'
        '<ul class="workflow-activities"><li>Reviews system exceptions</li><li>Records the human decision</li><li>Sets final loan status</li></ul>'
        '</div></div>'
    ) if show_deal_officer else ""
    return (
        f'<div class="workflow-graph{" has-officer" if show_deal_officer else ""}" role="status" aria-live="polite" '
        'aria-label="Orchestrator calls Extractor, then Agent 1 and Agent 2 run in parallel">'
        f'{stage_card("orchestrator")}'
        f'<div class="workflow-connector {states["orchestrator"]}"></div>'
        f'{stage_card("extractor")}'
        f'<div class="workflow-branch {branch_status}" aria-hidden="true">'
        '<span class="branch-in"></span><span class="branch-rail"></span>'
        '<span class="branch-top"></span><span class="branch-bottom"></span></div>'
        '<div><div class="workflow-parallel-label">Parallel execution</div><div class="workflow-parallel">'
        f'{stage_card("compliance_validator")}{stage_card("risk_summary")}'
        f'</div></div>{officer_card}</div>'
    )


def save_result(
    validation_id: str,
    filename: str,
    file_hash: str,
    result: dict,
    terms: dict,
    agent_metadata: dict,
    risk_result: dict,
) -> str:
    stamp = datetime.now(timezone.utc)
    automatic_decision = "Approve" if result["status"] == "COMPLIANT" else None
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT OR REPLACE INTO validations
            (validation_id, filename, file_hash, created_at, status, score, passed, warnings, failed, findings_json, terms_json, policy_filename, agent_metadata_json, risk_json, officer_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (validation_id, filename, file_hash, stamp.isoformat(), result["status"], result["score"], result["passed"], result["warnings"], result["failed"], json.dumps(result["findings"]), json.dumps(terms), None, json.dumps(agent_metadata), json.dumps(risk_result), automatic_decision))
    return validation_id


def load_history(search: str = "") -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT validation_id, filename, created_at, status, score, passed, warnings, failed, officer_decision, findings_json, terms_json, agent_metadata_json, risk_json FROM validations"
        params = ()
        if search:
            query += " WHERE lower(filename) LIKE ? OR lower(validation_id) LIKE ? OR lower(status) LIKE ?"
            token = f"%{search.lower()}%"
            params = (token, token, token)
        query += " ORDER BY created_at DESC"
        return pd.read_sql_query(query, conn, params=params)


def _execution_statuses(metadata: dict | None) -> dict[str, str]:
    metadata = metadata or {}
    recorded = metadata.get("execution_statuses", {})
    statuses = {}
    for stage in WORKFLOW_STAGES:
        value = str(recorded.get(stage, "SUCCESS")).upper().replace(" ", "_")
        statuses[stage] = value if value in {"SUCCESS", "FAILURE", "IN_PROGRESS", "NOT_RUN"} else "NOT_RUN"
    return statuses


def _status_label(status: str) -> str:
    return status.replace("_", " ").capitalize()


def _friendly_term_value(field: str, value, currency: str | None = None) -> str:
    if value in (None, ""):
        return "Not available"
    money_fields = {
        "loan_amount", "interest_amount", "total_debt", "tangible_net_worth", "current_assets",
        "current_liabilities", "collateral_market_value", "collateral_realisable_value",
        "total_facility_amount", "monthly_principal_instalment",
    }
    percent_fields = {"interest_rate_pct", "revenue_growth_pct", "stated_ltv"}
    ratio_fields = {"estimated_dscr", "debt_to_equity", "current_ratio", "stated_collateral_coverage"}
    if field in money_fields and isinstance(value, (int, float)):
        symbol = "₹" if str(currency).upper() == "INR" else f"{currency} " if currency else ""
        return f"{symbol}{value:,.0f}"
    if field in percent_fields and isinstance(value, (int, float)):
        return f"{value:,.2f}%"
    if field in ratio_fields and isinstance(value, (int, float)):
        return f"{value:,.2f}x"
    if field == "term_years":
        return f"{value} years"
    if field == "term_months":
        return f"{value} months"
    if field == "number_of_instalments":
        return f"{value} instalments"
    return str(value)


def _business_report_text(row, terms: dict, findings: list[dict], risk: dict, metadata: dict) -> str:
    lines = [
        "PREVIOUS DEAL VALIDATION REPORT",
        f"Validation ID: {row.validation_id}",
        f"Deal document: {row.filename}",
        f"Validated: {row.created_at}",
        f"Compliance outcome: {_status_label(str(row.status))}",
        f"Final loan approval status: {getattr(row, 'officer_decision', None) or 'Not decided'}",
        f"Overall risk: {_status_label(str(risk.get('overall_risk', 'UNKNOWN')))}",
        f"Recommendation: {_status_label(str(risk.get('recommendation', 'UNKNOWN')))}",
        f"Confidence: {100 * float(risk.get('confidence', 0)):.0f}%",
        "", "EXTRACTED DEAL TERMS",
    ]
    currency = terms.get("currency")
    for field, value in terms.items():
        lines.append(f"- {field.replace('_', ' ').title()}: {_friendly_term_value(field, value, currency)}")
    lines.extend(["", "COMPLIANCE FINDINGS"])
    for finding in findings:
        lines.append(f"- {finding.get('status', 'UNKNOWN')} — {finding.get('title', 'Finding')}: {finding.get('detail', '')}")
    lines.extend([
        "", "RISK ASSESSMENT",
        risk.get("summary", "No risk summary is available."),
        "", "EXECUTION INFORMATION",
        f"Extractor model: {metadata.get('model_name', 'Unknown')}",
        f"Compliance rules: {metadata.get('rule_catalog_version', 'Unknown')}",
        f"Risk rules: {metadata.get('risk_rule_catalog_version', 'Unknown')}",
    ])
    return "\n".join(lines)


def render_result(
    result: dict,
    validation_id: str,
    filename: str,
    terms: dict,
    metadata: dict | None = None,
    risk_result: dict | None = None,
) -> None:
    metadata = metadata or {}
    risk_result = risk_result or {}
    statuses = _execution_statuses(metadata)
    if result.get("status") == "NON_COMPLIANT":
        statuses["compliance_validator"] = "NON_COMPLIANT"
    elif result.get("status") == "INSUFFICIENT_DATA":
        statuses["compliance_validator"] = "INSUFFICIENT_DATA"
    else:
        statuses["compliance_validator"] = "COMPLIANT"
    if risk_result.get("overall_risk") == "LOW" and risk_result.get("recommendation") == "PROCEED":
        statuses["risk_summary"] = "ACCEPTABLE_RISK"
    elif risk_result.get("overall_risk"):
        statuses["risk_summary"] = "ADVERSE_RISK"
    extracted_count = sum(value not in (None, "") for value in terms.values())
    missing_count = len(terms) - extracted_count
    extractor_warnings = metadata.get("warnings", [])
    risk_flags = risk_result.get("risk_flags", [])
    risk_missing = risk_result.get("missing_information", [])
    summaries = {
        "orchestrator": "Extractor completed; Agent 1 and Agent 2 were coordinated in parallel.",
        "extractor": f"{extracted_count} of {len(terms)} fields extracted · {len(extractor_warnings)} warnings",
        "compliance_validator": f"{result['passed']} passed · {result['warnings']} insufficient · {result['failed']} failed",
        "risk_summary": f"{len(risk_flags)} risks flagged · {len(risk_missing)} missing inputs",
    }
    report = json.dumps({
        "validation_id": validation_id,
        "filename": filename,
        "execution": {"statuses": statuses, "metadata": metadata},
        "extracted_terms": terms,
        "agent_1_validation": result,
        "agent_2_risk_summary": risk_result,
    }, indent=2)

    risk_value = str(risk_result.get("overall_risk", "Unknown")).replace("_", " ").capitalize()
    recommendation_value = str(risk_result.get("recommendation", "Unknown")).replace("_", " ").capitalize()
    confidence_value = f"{100 * float(risk_result.get('confidence', 0)):.0f}%"
    overview_items = []
    for stage, (icon, name, _activities) in WORKFLOW_STAGES.items():
        stage_status = statuses[stage]
        overview_items.append(
            '<div class="overview-item"><div class="overview-icon">' + AGENT_ICONS[stage] + '</div><div>'
            f'<div class="overview-name">{html.escape(name)}</div>'
            f'<div class="overview-status {stage_status.lower()}">{html.escape(_status_label(stage_status))} · {html.escape(summaries[stage])}</div>'
            '</div></div>'
        )

    st.markdown('<div class="results-dashboard">', unsafe_allow_html=True)
    validation_col, overview_col, execution_col = st.columns(3, gap="small")
    with validation_col:
        with st.container(border=True, height=260):
            completed_value = str(metadata.get("completed_at", "Recorded result"))
            if "T" in completed_value:
                completed_value = completed_value.replace("T", " · ").replace("+00:00", " UTC")
            st.markdown(
                '<div class="validation-summary-head">✓ Validation complete</div>'
                '<div class="validation-summary-grid">'
                f'<div><div class="validation-summary-label">Validation ID</div><div class="validation-summary-value">{html.escape(validation_id)}</div></div>'
                f'<div><div class="validation-summary-label">Completed</div><div class="validation-summary-value">{html.escape(completed_value)}</div></div>'
                f'<div><div class="validation-summary-label">Deal document</div><div class="validation-summary-value">{html.escape(filename)}</div></div>'
                f'<div><div class="validation-summary-label">Status</div><div class="validation-summary-value success">Completed</div></div>'
                '</div>', unsafe_allow_html=True,
            )
            st.download_button("Download report", report, file_name=f"{validation_id}.json",
                               mime="application/json", width="stretch")
    with overview_col:
        with st.container(border=True, height=260):
            st.markdown('<div class="dashboard-panel-title">Execution overview <span class="dashboard-panel-note">4 components</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="overview-quad">{"".join(overview_items)}</div>', unsafe_allow_html=True)
    with execution_col:
        with st.container(border=True, height=260):
            st.markdown('<div class="dashboard-panel-title">Execution details <span class="dashboard-panel-note">Expanded</span></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="execution-detail-grid">'
                f'<div class="execution-detail-item"><div class="execution-detail-name">Orchestrator</div><div class="execution-detail-copy">{html.escape(_status_label(statuses["orchestrator"]))} · Parallel handoff completed</div></div>'
                f'<div class="execution-detail-item"><div class="execution-detail-name">Extractor</div><div class="execution-detail-copy">{extracted_count}/{len(terms)} fields · {len(extractor_warnings)} warnings · {html.escape(str(metadata.get("model_name", "Unknown")))}</div></div>'
                f'<div class="execution-detail-item"><div class="execution-detail-name">Compliance Validator</div><div class="execution-detail-copy">{html.escape(str(result["status"]).replace("_", " ").capitalize())} · Rules {html.escape(str(metadata.get("rule_catalog_version", "Unknown")))}</div></div>'
                f'<div class="execution-detail-item"><div class="execution-detail-name">Risk &amp; Summary</div><div class="execution-detail-copy">{html.escape(risk_value)} · {html.escape(recommendation_value)} · {confidence_value}</div></div>'
                '</div>', unsafe_allow_html=True,
            )

    bottom_left, bottom_right = st.columns(2, gap="small")
    with bottom_left:
        with st.container(border=True, height=215):
            st.markdown('<div class="dashboard-panel-title">Review outcome <span class="dashboard-panel-note">At a glance</span></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="review-outcome-grid">'
                '<div class="review-outcome-card"><div class="review-outcome-title">Compliance Validator</div><div class="review-stat-grid">'
                f'<div><div class="review-stat-label">Score</div><div class="review-stat-value">{result["score"]}%</div></div>'
                f'<div><div class="review-stat-label">Passed</div><div class="review-stat-value">{result["passed"]}</div></div>'
                f'<div><div class="review-stat-label">Insufficient</div><div class="review-stat-value">{result["warnings"]}</div></div>'
                f'<div><div class="review-stat-label">Failed</div><div class="review-stat-value">{result["failed"]}</div></div></div></div>'
                '<div class="review-outcome-card"><div class="review-outcome-title">Risk &amp; Summary</div><div class="risk-outcome-row">'
                f'<div><div class="review-stat-label">Overall risk</div><div class="review-stat-value">{html.escape(risk_value)}</div></div>'
                f'<div><div class="review-stat-label">Recommendation</div><div class="review-stat-value">{html.escape(recommendation_value)}</div></div>'
                f'<div><div class="review-stat-label">Confidence</div><div class="review-stat-value">{confidence_value}</div></div></div>'
                f'<div class="review-summary-copy">{html.escape(risk_result.get("summary", "No risk summary is available."))}</div></div></div>',
                unsafe_allow_html=True,
            )
    with bottom_right:
        with st.container(border=True, height=215):
            st.markdown('<div class="dashboard-panel-title">Supporting details <span class="dashboard-panel-note">Open when needed</span></div>', unsafe_allow_html=True)
            detail_state_key = f"selected_detail_{validation_id}"
            with st.container(key=f"supporting_details_{validation_id}"):
                detail_top_left, detail_top_right = st.columns(2, gap="small")
                with detail_top_left:
                    if st.button("Extracted terms", key=f"open_terms_{validation_id}", icon=":material/data_object:", width="stretch"):
                        st.session_state[detail_state_key] = "Extracted terms"
                with detail_top_right:
                    if st.button("Compliance findings", key=f"open_compliance_{validation_id}", icon=":material/checklist:", width="stretch"):
                        st.session_state[detail_state_key] = "Compliance findings"
                detail_bottom_left, detail_bottom_right = st.columns(2, gap="small")
                with detail_bottom_left:
                    if st.button("Risk assessment", key=f"open_risk_{validation_id}", icon=":material/warning:", width="stretch"):
                        st.session_state[detail_state_key] = "Risk assessment"
                with detail_bottom_right:
                    if st.button("Additional insights", key=f"open_insights_{validation_id}", icon=":material/lightbulb:", width="stretch"):
                        st.session_state[detail_state_key] = "Additional insights"
            selected_detail = st.session_state.get(detail_state_key)
            if selected_detail is None:
                st.caption("Select an area to view its detailed evidence.")
    st.markdown('</div>', unsafe_allow_html=True)
    if selected_detail == "Compliance findings":
        finding_rows = [{
            "Rule": finding["title"],
            "Status": finding["status"],
            "Original value": str(finding.get("original_value")) if finding.get("original_value") is not None else "Not found",
            "Threshold value": str(finding.get("threshold_value")) if finding.get("threshold_value") is not None else "Not configured",
            "Details": finding["detail"],
        } for finding in result["findings"]]
        st.dataframe(pd.DataFrame(finding_rows), width="stretch", hide_index=True)
    elif selected_detail == "Risk assessment":
        missing_items = risk_result.get("missing_information", [])
        positive_items = risk_result.get("positive_factors", [])
        assessment_status = "Unable to assess fully" if risk_value == "Unable to assess" else f"{risk_value} risk"
        assessment_context = (
            f"{len(missing_items)} critical input(s) are missing. " if missing_items else "All critical configured inputs were available. "
        ) + risk_result.get("summary", "No risk summary is available.")
        risk_items_html = "".join(
            f'<li><strong>{html.escape(str(flag["title"]))}</strong> · {html.escape(str(flag["severity"]).title())}<br>'
            f'{html.escape(str(flag.get("recommended_action", "Review the finding.")))}</li>'
            for flag in risk_flags
        ) or '<li>No threshold breaches were identified in the available financial data.</li>'
        positive_items_html = "".join(
            f'<li>{html.escape(str(factor))}</li>' for factor in positive_items
        ) or '<li>No positive factors were available from the configured assessment.</li>'
        st.markdown(
            '<div class="risk-detail-pane"><div class="risk-detail-heading">Risk assessment '
            '<span>All sections visible together</span></div><div class="risk-detail-grid">'
            '<div class="risk-detail-card"><div class="risk-detail-title">Assessment status</div>'
            f'<div class="risk-detail-status">{html.escape(assessment_status)}</div>'
            f'<div class="risk-detail-copy">{html.escape(assessment_context)}</div></div>'
            '<div class="risk-detail-card"><div class="risk-detail-title">Risk findings</div>'
            f'<div class="risk-detail-scroll"><ul class="risk-detail-list">{risk_items_html}</ul></div></div>'
            f'<div class="risk-detail-card"><div class="risk-detail-title">Positive factors · {len(positive_items)}</div>'
            f'<div class="risk-detail-scroll"><ul class="risk-detail-list">{positive_items_html}</ul></div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    elif selected_detail == "Extracted terms":
        term_rows = [{
            "Field": key.replace("_", " ").title(),
            "Extracted value": str(value) if value not in (None, "") else "Not found",
        } for key, value in terms.items()]
        st.dataframe(pd.DataFrame(term_rows), width="stretch", hide_index=True)
    elif selected_detail == "Additional insights":
        st.info("No additional insights are available in the current prototype. This section is reserved for future agents or review outputs.")


inject_styles()
init_db()
st.markdown(hero_markup(), unsafe_allow_html=True)

new_tab, current_tab, history_tab = st.tabs(["New validation", "Current Validation Results", "Validation history"])

with new_tab:
    st.markdown('<div class="section-title">New deal validation</div><div class="section-note">Upload one text-based PDF or TXT deal document up to 25 MB.</div>', unsafe_allow_html=True)
    upload_col, conditions_col, evaluation_col = st.columns([.75, 1.0, 1.55], gap="small")
    with upload_col:
        with st.container(border=True, height=250, key="new_validation_upload"):
            st.markdown('<div class="new-validation-panel-title">Upload deal document</div><div class="new-validation-panel-note">PDF or TXT · Maximum 25 MB</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload deal document", type=["pdf", "txt"], label_visibility="collapsed",
                                        help="The extraction precursor converts this document into structured deal terms.")
            run = st.button("Validate deal", type="primary", disabled=uploaded is None, width="stretch")
    matching_result = bool(
        uploaded is not None
        and st.session_state.get("current_filename") == uploaded.name
        and st.session_state.get("current_result")
    )
    displayed_result = st.session_state.get("current_result") if matching_result else None
    displayed_risk = st.session_state.get("current_risk_result") if matching_result else None
    displayed_metadata = st.session_state.get("current_agent_metadata", {}) if matching_result else {}
    displayed_validation_id = st.session_state.get("current_validation_id") if matching_result else None
    officer_required = requires_deal_officer(displayed_result, displayed_risk)
    displayed_thread_id = displayed_metadata.get("langgraph_thread_id")
    checkpoint_ready = bool(
        officer_required
        and displayed_metadata.get("langgraph_workflow_status") == "INTERRUPTED"
        and displayed_thread_id
        and DealReviewOrchestrator(
            APP_DIR / "data", APP_DIR / "rules.json", APP_DIR / "risk_rules.json"
        ).has_pending_interrupt(displayed_thread_id)
    )
    officer_decision_key = f"officer_decision_{displayed_validation_id}" if displayed_validation_id else "officer_decision_pending"
    if displayed_validation_id and officer_decision_key not in st.session_state:
        st.session_state[officer_decision_key] = load_officer_decision(displayed_validation_id)
    officer_decision = st.session_state.get(officer_decision_key)

    with conditions_col:
        with st.container(border=True, height=250, key="new_validation_conditions"):
            conditions_slot = st.empty()
            conditions_slot.markdown(deal_conditions_markup(displayed_result, displayed_risk), unsafe_allow_html=True)
    with evaluation_col:
        with st.container(border=True, height=250, key="new_validation_evaluation"):
            system_col, officer_col = st.columns(2, gap="small")
            with system_col:
                evaluation_slot = st.empty()
                evaluation_slot.markdown(final_evaluation_markup(displayed_result, displayed_risk), unsafe_allow_html=True)
            with officer_col:
                with st.container(key="deal_officer_panel"):
                    st.markdown('<div class="new-validation-panel-title">Deal Officer Validation</div>', unsafe_allow_html=True)
                    if checkpoint_ready:
                        officer_note = "Review paused. Examine the completed outputs and record a decision."
                    elif officer_required:
                        officer_note = "Human review is required, but no resumable workflow checkpoint is available."
                    else:
                        officer_note = "Available only after a completed automated review returns an adverse outcome."
                    st.markdown(f'<div class="new-validation-panel-note">{html.escape(officer_note)}</div>', unsafe_allow_html=True)
                    actions_enabled = checkpoint_ready and not officer_decision
                    action_approve, action_reject, action_hold = st.columns(3, gap="small")
                    with action_approve:
                        approve_clicked = st.button("Approve", disabled=not actions_enabled, key=f"officer_action_approve_{displayed_validation_id}", width="stretch")
                    with action_reject:
                        reject_clicked = st.button("Reject", disabled=not actions_enabled, key=f"officer_action_reject_{displayed_validation_id}", width="stretch")
                    with action_hold:
                        hold_clicked = st.button("On Hold", disabled=not actions_enabled, key=f"officer_action_hold_{displayed_validation_id}", width="stretch")
                    selected_decision = "Approve" if approve_clicked else "Reject" if reject_clicked else "On Hold" if hold_clicked else None
                    if selected_decision and displayed_validation_id:
                        thread_id = displayed_thread_id
                        if checkpoint_ready:
                            try:
                                resumed = DealReviewOrchestrator(
                                    APP_DIR / "data", APP_DIR / "rules.json", APP_DIR / "risk_rules.json"
                                ).resume(thread_id, selected_decision)
                                displayed_metadata["langgraph_workflow_status"] = resumed.workflow_status
                                displayed_metadata["langgraph_final_decision"] = resumed.final_decision
                                st.session_state.current_agent_metadata = displayed_metadata
                            except Exception as exc:
                                st.error(f"The paused LangGraph review could not be resumed: {exc}")
                                selected_decision = None
                        if selected_decision:
                            st.session_state[officer_decision_key] = selected_decision
                            officer_decision = selected_decision
                            save_officer_decision(displayed_validation_id, selected_decision, displayed_metadata)
                    decision_text = officer_decision or (
                        "Awaiting deal officer decision" if checkpoint_ready
                        else "Checkpoint unavailable" if officer_required
                        else "Not required"
                    )
                    decision_class = {
                        "Approve": "approve", "Reject": "reject", "On Hold": "on-hold",
                    }.get(officer_decision, "active" if checkpoint_ready else "")
                    st.markdown(
                        f'<div class="final-evaluation-label">Final Loan Approval Status</div>'
                        f'<div class="officer-status {decision_class}">{html.escape(decision_text)}</div>',
                        unsafe_allow_html=True,
                    )

    workflow_states = {stage: "idle" for stage in WORKFLOW_STAGES}
    if displayed_result and displayed_risk:
        orchestrator_state = (
            "interrupted"
            if displayed_metadata.get("langgraph_workflow_status") == "INTERRUPTED" and not officer_decision
            else "completed"
        )
        workflow_states.update({"orchestrator": orchestrator_state, "extractor": "completed"})
        workflow_states["compliance_validator"] = {
            "COMPLIANT": "compliant", "NON_COMPLIANT": "non_compliant",
            "INSUFFICIENT_DATA": "insufficient_data",
        }.get(displayed_result.get("status"), "failed")
        workflow_states["risk_summary"] = (
            "acceptable_risk"
            if displayed_risk.get("overall_risk") == "LOW" and displayed_risk.get("recommendation") == "PROCEED"
            else "adverse_risk"
        )
    mode = "Ollama LLM"
    model_name = os.getenv("EXTRACTOR_MODEL", "gemma3:4b")
    execution_status = st.empty()

    def update_execution_status(message: str, status: str = "idle") -> None:
        execution_status.markdown(
            f'<div class="execution-status {status}">{html.escape(message)}</div>',
            unsafe_allow_html=True,
        )

    selected_note = (
        st.session_state.get("last_execution_message")
        if matching_result and st.session_state.get("last_execution_message")
        else f"Ready · {uploaded.name} · {len(uploaded.getvalue()) / 1024:.1f} KB"
        if uploaded else "Upload a deal document to begin."
    )
    update_execution_status(selected_note)

    st.markdown('<div class="workflow-heading">Review workflow</div><div class="workflow-note">Live execution status and responsibility of each stage</div>', unsafe_allow_html=True)
    workflow_slot = st.empty()

    def update_workflow(stage: str | None = None, status: str | None = None) -> None:
        if stage == "parallel_agents" and status == "in_progress":
            workflow_states["compliance_validator"] = "in_progress"
            workflow_states["risk_summary"] = "in_progress"
        elif stage and status:
            workflow_states[stage] = status
        # A red or failed specialist does not by itself mean human review is
        # resumable. Show the human node only after LangGraph has checkpointed
        # an actual interrupt (or after a decision was recorded).
        show_officer_step = checkpoint_ready or bool(officer_decision)
        workflow_slot.markdown(
            workflow_timeline(workflow_states, show_officer_step, officer_decision),
            unsafe_allow_html=True,
        )
        compliance_status = workflow_states["compliance_validator"]
        risk_status = workflow_states["risk_summary"]
        finished_states = {"completed", "compliant", "non_compliant", "insufficient_data", "acceptable_risk", "adverse_risk"}
        if compliance_status == "in_progress" and risk_status == "in_progress":
            update_execution_status("Agent 1 and Agent 2 are running in parallel…", "in_progress")
        elif compliance_status in finished_states and risk_status == "in_progress":
            update_execution_status("Agent 1 completed; Agent 2 is still running…", "in_progress")
        elif risk_status in finished_states and compliance_status == "in_progress":
            update_execution_status("Agent 2 completed; Agent 1 is still running…", "in_progress")
        elif stage and status == "in_progress":
            messages = {
                "orchestrator": "Starting the coordinated deal review…",
                "extractor": "Orchestrator called the Extractor; extracting structured terms…",
                "compliance_validator": "Running Agent 1 compliance validation…",
                "risk_summary": "Agent 2 is flagging risks and generating the grounded summary…",
            }
            update_execution_status(messages[stage], "in_progress")
        elif stage and status == "failed":
            update_execution_status(f"{WORKFLOW_STAGES[stage][1]} failed.", "failed")
        elif stage == "orchestrator" and status == "interrupted":
            update_execution_status("Review paused · waiting for the Deal Officer Validator decision.", "in_progress")

    update_workflow()
    if run and uploaded:
        progress_events: list[tuple[str, str]] = []
        progress_lock = threading.Lock()

        def collect_progress(stage: str, status: str) -> None:
            """Collect LangGraph events without calling Streamlit from worker threads."""
            with progress_lock:
                progress_events.append((stage, status))

        def render_collected_progress() -> None:
            """Render collected events only from Streamlit's main script thread."""
            with progress_lock:
                events = list(progress_events)
                progress_events.clear()
            for event_stage, event_status in events:
                update_workflow(event_stage, event_status)

        try:
            evaluation_slot.markdown(final_evaluation_markup(state="in_progress"), unsafe_allow_html=True)
            conditions_slot.markdown(deal_conditions_markup(), unsafe_allow_html=True)
            update_workflow("orchestrator", "in_progress")
            text = extract_text(uploaded)
            if not text.strip():
                raise ValueError("No readable text was found. The PDF may be scanned and require OCR.")
            flow = DealReviewOrchestrator(
                APP_DIR / "data", APP_DIR / "rules.json", APP_DIR / "risk_rules.json"
            ).run(
                document_text=text, document_bytes=uploaded.getvalue(), mode=mode, model_name=model_name,
                progress_callback=collect_progress,
            )
            render_collected_progress()
            terms = flow.extractor_output.terms.model_dump(mode="json")
            validation = flow.agent1_result
            risk_result = flow.agent2_result.model_dump(mode="json")
            findings = []
            for item in validation.findings:
                css = "pass" if item.status == "PASS" else "fail" if item.status == "FAIL" else "warn"
                findings.append({
                    "title": f"{item.rule_id} · {item.field.replace('_', ' ').title()}",
                    "status": item.status,
                    "severity": css,
                    "original_value": item.actual_value,
                    "threshold_value": item.threshold,
                    "detail": f"{item.message} Actual: {item.actual_value!r}; operator: {item.operator}; threshold: {item.threshold!r}.",
                })
            result = {"status": validation.overall_status, "score": validation.score,
                      "passed": validation.passed, "warnings": validation.insufficient_data,
                      "failed": validation.failed, "findings": findings}
            if validation.overall_status == "NON_COMPLIANT":
                update_workflow("compliance_validator", "non_compliant")
            elif validation.overall_status == "INSUFFICIENT_DATA":
                update_workflow("compliance_validator", "insufficient_data")
            digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
            agent_metadata = {
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                "extraction_method": flow.extractor_output.extraction_method,
                "model_name": flow.extractor_output.model_name,
                "warnings": flow.extractor_output.warnings,
                "agent1_name": validation.agent_name,
                "rule_catalog_version": validation.rule_catalog_version,
                "agent2_name": flow.agent2_result.agent_name,
                "risk_rule_catalog_version": flow.agent2_result.risk_rule_catalog_version,
                "summary_model": flow.agent2_result.summary_model,
                "extractor_handoff_path": flow.extractor_handoff_path,
                "agent2_output_path": flow.agent2_output_path,
                "langgraph_thread_id": flow.thread_id,
                "langgraph_workflow_status": flow.workflow_status,
                "langgraph_interrupt_payload": flow.interrupt_payload,
                "execution_statuses": {
                    stage: {
                        "completed": "SUCCESS",
                        "failed": "FAILURE",
                        "in_progress": "IN_PROGRESS",
                        "idle": "NOT_RUN",
                        "non_compliant": "SUCCESS",
                        "insufficient_data": "SUCCESS",
                        "compliant": "SUCCESS",
                        "acceptable_risk": "SUCCESS",
                        "adverse_risk": "SUCCESS",
                        "interrupted": "IN_PROGRESS",
                    }[status]
                    for stage, status in workflow_states.items()
                },
            }
            validation_id = save_result(
                flow.case_id, uploaded.name, digest, result, terms, agent_metadata, risk_result
            )
            st.session_state.current_result = result
            st.session_state.current_terms = terms
            st.session_state.current_agent_metadata = agent_metadata
            st.session_state.current_risk_result = risk_result
            st.session_state.current_validation_id = validation_id
            st.session_state.current_filename = uploaded.name
            st.session_state.pop(f"officer_decision_{validation_id}", None)
            st.session_state.last_execution_message = (
                f"Validation complete · {validation_id} · Risk: {flow.agent2_result.overall_risk} · "
                f"Recommendation: {flow.agent2_result.recommendation}"
            )
            st.rerun()
        except Exception as exc:
            render_collected_progress()
            evaluation_slot.markdown(
                '<div class="new-validation-panel-title">System Validation</div>'
                '<div class="new-validation-panel-note">Policy Validation Center Review Comments</div>'
                '<div class="final-evaluation-status pending">● Evaluation failed</div>',
                unsafe_allow_html=True,
            )
            for stage_name, stage_status in workflow_states.items():
                if stage_status == "in_progress":
                    update_workflow(stage_name, "failed")
            if workflow_states["orchestrator"] != "failed":
                update_workflow("orchestrator", "failed")
            error_detail = str(exc).strip() or exc.__class__.__name__
            update_execution_status(f"Validation could not be completed: {error_detail}", "failed")

with current_tab:
    if "current_result" in st.session_state:
        render_result(
            st.session_state.current_result,
            st.session_state.current_validation_id,
            st.session_state.current_filename,
            st.session_state.current_terms,
            st.session_state.get("current_agent_metadata"),
            st.session_state.get("current_risk_result"),
        )
    else:
        latest = load_history().head(1)
        if latest.empty:
            st.info("No validation has been run yet. Upload a deal document in New validation to begin.")
        else:
            row = latest.iloc[0]
            result = {"status": row.status, "score": int(row.score), "passed": int(row.passed), "warnings": int(row.warnings), "failed": int(row.failed), "findings": json.loads(row.findings_json)}
            render_result(
                result, row.validation_id, row.filename, json.loads(row.terms_json),
                json.loads(row.agent_metadata_json), json.loads(row.risk_json),
            )

with history_tab:
    st.markdown('<div class="section-title">Previous validations</div><div class="section-note">Search, filter and reopen earlier policy checks.</div>', unsafe_allow_html=True)
    query = st.text_input("Search history", placeholder="Search by file name, validation ID or status")
    history = load_history(query)
    if history.empty:
        st.info("No matching validation records found.")
    else:
        display = history[["validation_id", "filename", "created_at", "status", "officer_decision", "score", "passed", "warnings", "failed"]].copy()
        display["officer_decision"] = display["officer_decision"].fillna("Not decided")
        display["created_at"] = pd.to_datetime(display["created_at"], utc=True).dt.strftime("%d %b %Y · %H:%M UTC")
        display.columns = ["Validation ID", "Deal document", "Validated", "Status", "Final loan approval status", "Score", "Passed", "Insufficient", "Failed"]
        st.dataframe(display, width="stretch", hide_index=True)
        selected_id = st.selectbox("Open a previous validation", history["validation_id"].tolist())
        if selected_id:
            row = history.loc[history.validation_id == selected_id].iloc[0]
            terms = json.loads(row.terms_json)
            findings = json.loads(row.findings_json)
            metadata = json.loads(row.agent_metadata_json)
            risk = json.loads(row.risk_json)
            risk_label = _status_label(str(risk.get("overall_risk", "UNKNOWN")))
            recommendation_label = _status_label(str(risk.get("recommendation", "UNKNOWN")))
            confidence = f"{100 * float(risk.get('confidence', 0)):.0f}%"
            final_approval_status = row.officer_decision or "Not decided"
            validated_at = pd.to_datetime(row.created_at, utc=True).strftime("%d %b %Y · %H:%M UTC")
            technical_report = json.dumps({
                "validation_id": row.validation_id, "filename": row.filename,
                "created_at": row.created_at, "extracted_terms": terms,
                "compliance_findings": findings, "risk_assessment": risk,
                "execution_metadata": metadata, "final_loan_approval_status": final_approval_status,
            }, indent=2)
            business_report = _business_report_text(row, terms, findings, risk, metadata)

            st.markdown('<div class="section-title">Previous validation review</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="history-review-head">'
                f'<div class="history-review-stat"><div class="history-review-label">Validation ID</div><div class="history-review-value">{html.escape(str(row.validation_id))}</div></div>'
                f'<div class="history-review-stat"><div class="history-review-label">Validated</div><div class="history-review-value">{html.escape(validated_at)}</div></div>'
                f'<div class="history-review-stat"><div class="history-review-label">Compliance</div><div class="history-review-value">{html.escape(_status_label(str(row.status)))}</div></div>'
                f'<div class="history-review-stat"><div class="history-review-label">Overall risk</div><div class="history-review-value">{html.escape(risk_label)}</div></div>'
                f'<div class="history-review-stat"><div class="history-review-label">Recommendation</div><div class="history-review-value">{html.escape(recommendation_label)}</div></div>'
                f'<div class="history-review-stat"><div class="history-review-label">Confidence</div><div class="history-review-value">{confidence}</div></div>'
                f'<div class="history-review-stat"><div class="history-review-label">Final loan approval</div><div class="history-review-value">{html.escape(final_approval_status)}</div></div>'
                '</div>', unsafe_allow_html=True,
            )
            report_col, json_col, spacer_col = st.columns([1, 1, 3], gap="small")
            with report_col:
                st.download_button("Download report", business_report,
                                   file_name=f"{row.validation_id}_report.txt", mime="text/plain", width="stretch")
            with json_col:
                st.download_button("Technical JSON", technical_report,
                                   file_name=f"{row.validation_id}.json", mime="application/json", width="stretch")

            terms_tab, compliance_tab, risk_tab, execution_tab = st.tabs(
                ["Deal terms", "Compliance findings", "Risk assessment", "Execution details"]
            )
            with terms_tab:
                currency = terms.get("currency")
                term_cards = "".join(
                    '<div class="history-term">'
                    f'<div class="history-term-label">{html.escape(field.replace("_", " ").title())}</div>'
                    f'<div class="history-term-value">{html.escape(_friendly_term_value(field, value, currency))}</div></div>'
                    for field, value in terms.items()
                )
                st.markdown(f'<div class="history-term-grid">{term_cards}</div>', unsafe_allow_html=True)
            with compliance_tab:
                for finding in findings:
                    status = str(finding.get("status", "UNKNOWN"))
                    css = "pass" if status == "PASS" else "fail" if status == "FAIL" else "warn"
                    actual = finding.get("original_value")
                    threshold = finding.get("threshold_value")
                    values = ""
                    if actual is not None or threshold is not None:
                        values = f"Extracted value: {actual if actual is not None else 'Not available'} · Threshold: {threshold if threshold is not None else 'Not configured'}"
                    st.markdown(
                        f'<div class="history-finding {css}"><div class="history-finding-title">{html.escape(status)} — {html.escape(str(finding.get("title", "Finding")))}</div>'
                        f'<div class="history-finding-copy">{html.escape(values)}{("<br>" if values else "")}{html.escape(str(finding.get("detail", "")))}</div></div>',
                        unsafe_allow_html=True,
                    )
            with risk_tab:
                st.markdown(
                    f"**Overall risk:** {risk_label}  \n"
                    f"**Recommendation:** {recommendation_label}  \n"
                    f"**Confidence:** {confidence}  \n\n"
                    f"{risk.get('summary', 'No risk summary is available.')}"
                )
                if risk.get("risk_flags"):
                    st.markdown("**Risk findings**")
                    for flag in risk["risk_flags"]:
                        st.markdown(f"- **{flag.get('severity', 'Risk')} — {flag.get('title', 'Finding')}**: {flag.get('message', flag.get('detail', ''))}")
                if risk.get("positive_factors"):
                    st.markdown("**Positive factors**")
                    for factor in risk["positive_factors"]:
                        st.markdown(f"- {factor}")
                if risk.get("missing_information"):
                    st.markdown("**Missing information**")
                    for item in risk["missing_information"]:
                        st.markdown(f"- {item}")
            with execution_tab:
                execution_rows = [
                    {"Component": "Orchestrator", "Status": "Completed", "Details": "Coordinated extraction and parallel specialist execution"},
                    {"Component": "Extractor", "Status": "Completed", "Details": f"Model: {metadata.get('model_name', 'Unknown')}"},
                    {"Component": "Compliance Validator", "Status": _status_label(str(row.status)), "Details": f"Rules: {metadata.get('rule_catalog_version', 'Unknown')}"},
                    {"Component": "Risk & Summary", "Status": risk_label, "Details": f"Rules: {metadata.get('risk_rule_catalog_version', 'Unknown')}"},
                ]
                st.dataframe(pd.DataFrame(execution_rows), width="stretch", hide_index=True)
