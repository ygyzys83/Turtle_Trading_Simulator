from __future__ import annotations


CHART_COLORS = {
    "price": "#D8E0E9",
    "trend": "#5B8FF9",
    "entry": "#35C46A",
    "exit": "#E3AA42",
    "sell": "#FF6262",
    "atr": "#45B8A8",
    "grid": "#24364A",
    "text": "#91A1B4",
    "surface": "#111E2B",
    "border": "#2A4055",
}


TRADING_CONSOLE_CSS = """
<style>
    :root {
        --app-bg: #0B1420;
        --sidebar-bg: #0D1824;
        --surface: #111E2B;
        --surface-raised: #17283A;
        --surface-hover: #1D3247;
        --border: #263B4F;
        --border-strong: #35516B;
        --text: #E7EEF6;
        --text-secondary: #A6B5C5;
        --text-muted: #778A9E;
        --blue: #5B8FF9;
        --blue-deep: #2766C7;
        --green: #35C46A;
        --red: #FF6262;
        --amber: #E3AA42;
        --teal: #45B8A8;
        --radius: 6px;
    }

    html, body, .stApp, button, input, textarea, select {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing: 0;
    }

    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: "liga" !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--app-bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        background: transparent;
        border: 0;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1700px;
        padding: 0.5rem 1.75rem 3rem;
    }

    [data-testid="stLayoutWrapper"]:has(> .st-key-top_navigation) {
        position: sticky;
        top: 0;
        z-index: 900;
    }

    .st-key-top_navigation {
        position: relative;
        background: rgba(11, 20, 32, 0.98);
        border-bottom: 1px solid var(--border);
        padding: 0.45rem 0 0;
        margin: 0 0 0.55rem;
        gap: 0.35rem !important;
        backdrop-filter: blur(10px);
    }

    .st-key-top_navigation h1 {
        font-size: 1.05rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0 !important;
    }

    .st-key-top_navigation [data-testid="stMarkdownContainer"]:has(h1) {
        margin-bottom: 0 !important;
    }

    .st-key-top_navigation [role="radiogroup"] {
        margin-bottom: 0 !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        letter-spacing: 0 !important;
    }

    h1 {
        color: var(--text) !important;
        font-size: 1.48rem !important;
        line-height: 1.25 !important;
        font-weight: 650 !important;
        margin: 0 0 0.2rem !important;
        padding: 0 !important;
    }

    h2 {
        color: var(--text) !important;
        font-size: 1.12rem !important;
        line-height: 1.35 !important;
        font-weight: 640 !important;
        margin: 1.55rem 0 0.35rem !important;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    h3 {
        color: var(--text) !important;
        font-size: 0.94rem !important;
        line-height: 1.4 !important;
        font-weight: 620 !important;
        margin: 1.05rem 0 0.3rem !important;
    }

    h4 {
        color: var(--text) !important;
        font-size: 0.84rem !important;
        line-height: 1.4 !important;
        font-weight: 620 !important;
        margin: 0.9rem 0 0.3rem !important;
    }

    h5, h6 {
        color: var(--text-secondary) !important;
        font-size: 0.76rem !important;
        font-weight: 620 !important;
    }

    p, li, label, [data-testid="stCaptionContainer"] {
        color: var(--text-secondary);
        font-size: 0.84rem;
        line-height: 1.42;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 0.71rem;
        line-height: 1.45;
    }

    .ui-section-caption {
        color: var(--text-muted);
        font-size: 0.75rem;
        line-height: 1.45;
        margin-top: -0.15rem;
        margin-bottom: 0.95rem;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
        width: 288px !important;
        min-width: 288px !important;
        margin-left: 0 !important;
        transform: none !important;
        visibility: visible !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 0.7rem 0.75rem 1.5rem;
    }

    [data-testid="stSidebarHeader"] {
        display: none !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        display: none !important;
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 0.75rem !important;
    }

    [data-testid="stSidebar"] h3 {
        color: var(--text-secondary) !important;
        font-size: 0.70rem !important;
        font-weight: 680 !important;
        text-transform: uppercase;
        margin: 0.95rem 0 0.4rem !important;
        padding-top: 0.65rem;
        border-top: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .sidebar-kill-title {
        color: var(--red);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 0.25rem 0 0.1rem;
        border: 0;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        font-size: 0.74rem;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.32rem;
    }

    .metric-card {
        min-height: 78px;
        background: linear-gradient(180deg, rgba(23, 40, 58, 0.92), rgba(17, 30, 43, 0.92));
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.72rem 0.82rem;
        margin-bottom: 0.42rem;
        overflow: hidden;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.62rem;
        font-weight: 650;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 0.90rem;
        font-weight: 650;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .metric-sub {
        color: var(--text-muted);
        font-size: 0.64rem;
        line-height: 1.35;
        margin-top: 0.3rem;
    }

    .pos { color: var(--green) !important; }
    .neg { color: var(--red) !important; }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
        background: linear-gradient(180deg, rgba(23, 40, 58, 0.96), rgba(17, 30, 43, 0.96));
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin: 0.55rem 0 0.9rem;
        overflow: hidden;
    }

    .status-strip-item {
        min-width: 0;
        padding: 0.62rem 0.78rem;
        border-right: 1px solid var(--border);
    }

    .status-strip-item:last-child { border-right: 0; }
    .status-strip-label {
        color: var(--text-muted);
        font-size: 0.59rem;
        font-weight: 680;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
    }
    .status-strip-value {
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 640;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .status-strip-state {
        color: var(--text-muted);
        font-size: 0.60rem;
        margin-top: 0.2rem;
    }
    .status-positive .status-strip-value { color: var(--green); }
    .status-negative .status-strip-value { color: var(--red); }
    .status-warning .status-strip-value { color: var(--amber); }

    .rule-box {
        border-left: 2px solid var(--border-strong);
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .signal-long, .signal-exit, .signal-flat {
        display: inline-block;
        border-radius: 4px;
        padding: 0.42rem 0.7rem;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .signal-long { background: rgba(53, 196, 106, 0.12); color: var(--green); border: 1px solid rgba(53, 196, 106, 0.30); }
    .signal-exit { background: rgba(255, 98, 98, 0.12); color: var(--red); border: 1px solid rgba(255, 98, 98, 0.30); }
    .signal-flat { background: var(--surface); color: var(--text-secondary); border: 1px solid var(--border); }

    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        min-height: 2rem;
        border-radius: 4px !important;
        border: 1px solid var(--border-strong) !important;
        background: #183047 !important;
        color: var(--text) !important;
        font-size: 0.73rem !important;
        font-weight: 620 !important;
        box-shadow: none !important;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #21415D !important;
        border-color: #527493 !important;
    }

    [data-testid="baseButton-primary"],
    button[kind="primary"] {
        background: #1F7A45 !important;
        border-color: #35C46A !important;
        color: #FFFFFF !important;
    }

    button:disabled {
        opacity: 0.42 !important;
        cursor: not-allowed !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #101F2D !important;
        border-color: var(--border-strong) !important;
        border-radius: 4px !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }

    [data-baseweb="select"] > div:hover,
    [data-baseweb="input"] > div:hover {
        border-color: #526171 !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--blue) !important;
        border-color: var(--blue) !important;
    }

    [data-testid="stCheckbox"] label span:first-child {
        border-radius: 3px !important;
    }

    [data-testid="stExpander"] {
        background: rgba(17, 30, 43, 0.72);
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] summary {
        min-height: 2.25rem;
        color: var(--text-secondary);
        font-size: 0.74rem;
        font-weight: 610;
    }

    [data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }

    [data-testid="stDataFrame"] canvas,
    [data-testid="stTable"] {
        font-size: 0.76rem !important;
    }

    [data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--blue) !important;
        border-radius: var(--radius) !important;
        color: var(--text-secondary) !important;
    }

    [data-testid="stAlert"] p {
        font-size: 0.76rem;
        line-height: 1.5;
    }

    [data-baseweb="tab-list"] {
        gap: 1.2rem;
        border-bottom: 1px solid var(--border);
    }

    [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-size: 0.68rem !important;
        font-weight: 620 !important;
        padding: 0.45rem 0.1rem !important;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: var(--text) !important;
    }

    [data-testid="stMainBlockContainer"] [role="radiogroup"] {
        gap: 1.25rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.75rem;
    }

    [data-testid="stMainBlockContainer"] [role="radiogroup"] label {
        background: transparent !important;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.45rem 0.1rem 0.5rem !important;
        color: var(--text-muted) !important;
    }

    [data-testid="stMainBlockContainer"] [role="radiogroup"] label:has(input:checked) {
        border-bottom-color: var(--blue) !important;
        color: var(--text) !important;
    }

    [data-testid="stMainBlockContainer"] [role="radiogroup"] input,
    [data-testid="stMainBlockContainer"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.25rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        padding: 0.38rem 0.5rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        border-color: rgba(91, 143, 249, 0.65) !important;
        background: rgba(91, 143, 249, 0.10) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        min-height: 2.05rem;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        min-height: 2rem !important;
        font-size: 0.74rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(19, 34, 49, 0.88), rgba(14, 27, 40, 0.88));
        border-color: var(--border) !important;
        border-radius: var(--radius) !important;
    }

    [data-testid="stPlotlyChart"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }

    hr {
        border-color: var(--border) !important;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 3rem;
        }
        .status-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .status-strip-item {
            border-bottom: 1px solid var(--border);
        }
        .metric-card {
            min-height: 82px;
        }
        h1 { font-size: 1.4rem !important; }
        .st-key-top_navigation h1 { font-size: 1rem !important; }
        h2 { font-size: 1.15rem !important; }
    }
</style>
"""
