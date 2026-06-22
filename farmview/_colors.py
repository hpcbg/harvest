# Task phase → fill color
PHASE_COLOR = {
    "PENDING":     "#9DA8B7",
    "TRANSIT":     "#F59842",
    "EXECUTING":   "#3AB56E",
    "DONE":        "#1D9E75",
    "DELAYED":     "#F5C842",
    "INTERRUPTED": "#E05C5C",
}

PHASE_LABEL = {
    "PENDING":     "Pending",
    "TRANSIT":     "In Transit",
    "EXECUTING":   "Executing",
    "DONE":        "Done",
    "DELAYED":     "Delayed",
    "INTERRUPTED": "Interrupted",
}

# Task priority → edge color + linewidth
PRIORITY_EDGE_COLOR = {
    "urgent":   "#C0392B",
    "normal":   "#2E86C1",
    "flexible": "#95A5A6",
}

PRIORITY_EDGE_WIDTH = {
    "urgent":   2.5,
    "normal":   1.8,
    "flexible": 1.0,
}

# First word of task name → (matplotlib marker, long label)
TASK_TYPE_MARKER = {
    "mowing":     ("s", "Mowing"),
    "pruning":    ("p", "Pruning"),
    "transport":  ("D", "Transport"),
    "inspection": ("o", "Inspection"),
    "sprayer":    ("*", "Sprayer"),
    "tool":       ("^", "Tool Delivery"),
}
DEFAULT_MARKER = ("h", "Task")   # hexagon fallback

# Tractor status → body fill color
TRACTOR_STATUS_COLOR = {
    "idle":         "#7F8C8D",
    "charging":     "#1A6EBD",
    "transit":      "#E67E22",
    "executing":    "#27AE60",
    "offline":      "#922B21",
    "discharging":  "#1ABC9C",   # teal — V2L active, power flowing to farm loads
}

TRACTOR_STATUS_LABEL = {
    "idle":         "Idle",
    "charging":     "Charging",
    "transit":      "In Transit",
    "executing":    "Executing",
    "offline":      "Offline",
    "discharging":  "V2L Active",
}

# Battery SOC level colors
SOC_HIGH  = "#27AE60"   # > 60 %
SOC_MID   = "#F39C12"   # 30–60 %
SOC_LOW   = "#E74C3C"   # < 30 %

# Farm map background
FARM_BG       = "#E8F5E3"
DEPOT_BG      = "#FDF6EC"
CHARGER_COLOR = "#1A5276"

# ── MARL visualization ────────────────────────────────────────────────────────

# Tractor agent: 0 = IDLE (light gray), 1 = REQUEST_CHARGE (blue)
MARL_TRACTOR_CMAP = ["#D5D8DC", "#2E86C1"]

# Charger agent: 0 = OFF (near-white), 1 = LOW (teal-light), 2 = FULL (teal-dark)
MARL_CHARGER_CMAP = ["#F0F3F4", "#76D7C4", "#1A7A6E"]

# Load agent: -1→0 = not-scheduled (gray), 0→1 = ON (slate), 1→2 = shed (amber)
# Values in MARLStepLog are -1/0/1; add 1 before passing to imshow (range 0-2).
MARL_LOAD_CMAP = ["#D5D8DC", "#5D6D7E", "#F39C12"]

# Dynamic event type → annotation color
EVENT_COLORS = {
    "task_inject":    "#8E44AD",   # purple
    "tractor_offline": "#922B21",  # dark red
    "grid_reduce":    "#E67E22",   # orange
    "grid_restore":   "#1A7A6E",   # teal
    "power_outage":   "#C0392B",   # red
}

# Reward component colors
REWARD_COST_COLOR = "#E74C3C"
REWARD_PEAK_COLOR = "#F39C12"
REWARD_TASK_COLOR = "#27AE60"

# Tariff band backgrounds (for SOC chart backdrop)
TARIFF_VALLE_BG = "#E8F5E9"   # light green  — cheap night rate
TARIFF_LLANO_BG = "#FFFDE7"   # light yellow — mid rate
TARIFF_PUNTA_BG = "#FFEBEE"   # light red    — peak rate
