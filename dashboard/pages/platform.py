"""Platform: the service owner's console (customers, sign-up, the shared model, activity).

Only the platform owner sees this page. Customer administrators manage their own
organisation under Settings and cannot change anything here.
"""
from __future__ import annotations

import shutil

import pandas as pd
import streamlit as st

import config
import model_loader
from dashboard import database as storage
from dashboard import services
from dashboard import theme as ui
from evaluation import fmt_pct


def _lstm_model_inventory() -> None:
    """Display the depot LSTM models actually used by operational prediction.

    The former global XGBoost package is a historical benchmark, not an
    operational model. Do not expose its activation workflow as a way to
    replace the application's active LSTM forecasting models.
    """
    import depot_updates as dep

    st.subheader("Operational forecasting models · LSTM")
    st.caption(
        "Demand Prediction loads the selected depot's active Keras LSTM model "
        "and its paired scaler. The 30-day workflow uses its dedicated depot LSTM "
        "when that forecasting method is selected."
    )
    names = dep.visible_depots()
    if not names:
        st.info("No depots are registered. Add a depot in Depot Setup.")
        return

    rows = []
    for name in names:
        model, scaler = dep.active_paths(name)
        long_model = config.ROOT / "models" / "long_term_lstm" / name.lower().replace(" ", "_") / "model.keras"
        model_ready = model.is_file() and scaler.is_file()
        rows.append({
            "Depot": name,
            "Operational model": "LSTM" if model_ready else "Unavailable",
            "15-minute model": "Ready" if model_ready else "Model or scaler missing",
            "30-day dedicated LSTM": "Available" if long_model.is_file() else "Unavailable",
            "Active model file": model.name if model_ready else "—",
        })
    df = pd.DataFrame(rows)
    ready = int(df["15-minute model"].eq("Ready").sum())
    st.metric("Depots with active short-term LSTM", f"{ready} / {len(df)}")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.info(
        "XGBoost remains in the repository for historical benchmarking only. "
        "It is not activated by this page and is not used for the operational forecasts "
        "shown in Demand Prediction. To train or activate an LSTM, use Depot Setup."
    )


def _organisations() -> None:
    orgs = storage.list_organisations()
    customers = orgs[orgs["platform"] != 1]
    ui.cards([
        ui.card("Customer organisations", str(len(customers)), "", f"{int(customers['active'].sum())} active", icon_name="layers"),
        ui.card("User accounts", str(int(orgs["users"].sum())), "", "Across all organisations", icon_name="users"),
        ui.card("Depots", str(int(customers["depots"].fillna(0).sum())), "", "Configured by customers", icon_name="cube"),
        ui.card("Sign-up", "Open" if storage.signup_allowed() else "Closed", "", "Public Create account page", icon_name="shield"),
    ])
    notice = st.session_state.pop("platform_notice", None)
    if notice:
        st.success(notice, icon=":material/check_circle:")
    shown = pd.DataFrame({
        "Organisation": orgs["name"],
        "Type": orgs["platform"].map(lambda p: "Platform" if p == 1 else "Customer"),
        "Status": orgs["active"].map({1: "Active", 0: "Suspended"}),
        "Users": orgs["users"], "Administrators": orgs["admins"],
        "Depots": orgs["depots"].map(lambda d: "Not opened yet" if pd.isna(d) else str(int(d))),
        "Created": orgs["created_at"].str[:16], "Last sign-in": orgs["last_sign_in"].fillna("Never").str[:16],
    })
    st.dataframe(shown, hide_index=True, width="stretch")

    if customers.empty:
        ui.empty_state("No customers yet", "Organisations appear here when someone creates an account.")
        return
    names = customers.set_index("id")["name"].to_dict()
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    target = left.selectbox("Customer organisation", list(names), format_func=names.get)
    active = bool(customers.set_index("id").loc[target, "active"])
    if right.button("Suspend access" if active else "Reactivate access", width="stretch",
                    icon=":material/block:" if active else ":material/check_circle:"):
        try:
            storage.set_organisation_active(target, not active, services.user_email())
            st.session_state["platform_notice"] = f"{names[target]} {'suspended' if active else 'reactivated'}."
            st.rerun()
        except storage.AccountError as exc:
            st.error(str(exc), icon=":material/error:")
    ui.note("Suspending an organisation signs its users out and blocks sign-in until it is reactivated. "
            "Its data is kept.")


def _signup() -> None:
    allowed = storage.signup_allowed()
    choice = st.toggle("Allow new customers to create an account from the public page", value=allowed)
    if choice != allowed:
        storage.set_signup_allowed(choice, services.user_email())
        st.rerun()
    ui.note("When this is on, anyone can create a new organisation and becomes its administrator. They start with "
            "the sample depots and can upload their own data. When it is off, the Create account page is closed "
            "and only existing users can sign in.", title="What this controls")


def _activity() -> None:
    log = storage.read_platform_audit()
    if log.empty:
        ui.empty_state("No activity recorded yet", "Sign-ins, account changes and model changes appear here.")
        return
    st.dataframe(log.rename(columns={"created_at": "When", "organisation": "Organisation", "user": "User",
                                     "action": "Action", "detail": "Detail"}),
                 hide_index=True, width="stretch", height=min(560, 38 + 35 * len(log)))


def render() -> None:
    if not services.is_owner():
        st.error("This page is only available to the platform owner.", icon=":material/lock:")
        return
    ui.page_header("Platform", "Customers, sign-up, active depot LSTM models and activity",
                   config.PRODUCT_NAME, [("Role", "Platform owner"), ("Forecast model", "LSTM")])
    tabs = st.tabs(["Organisations", "Sign-up", "LSTM Models", "Activity"])
    with tabs[0]:
        _organisations()
    with tabs[1]:
        _signup()
    with tabs[2]:
        _lstm_model_inventory()
    with tabs[3]:
        _activity()
