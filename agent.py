# Agentic Login Interdiction with Multi-Agent LangGraph (Net Banking Use Case)
# Prerequisites: pip install langgraph fastapi uvicorn openai

from fastapi import FastAPI
from pydantic import BaseModel
import datetime
from langgraph.graph import StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt.tool_node import ToolNode
from typing import Dict, List
import random

app = FastAPI()

# --- 1. Request/Response Schemas ---
class LoginRequest(BaseModel):
    username: str
    ip_address: str
    device_id: str
    login_method: str  # web, mobile, voice, atm
    timestamp: str

class DecisionResponse(BaseModel):
    action: str
    explanation: str

# --- 2. Simulated User History DB ---
user_history = {
    "alice": {
        "last_ips": ["192.168.0.1"],
        "last_devices": ["device_123"],
        "last_login": "2025-06-11T09:00:00",
        "methods": ["web", "mobile"]
    }
}

# --- 3. IFM Risk Scoring Agent ---
def ifm_score(context: Dict) -> Dict:
    user = context["username"]
    ip = context["ip_address"]
    device = context["device_id"]
    method = context["login_method"]
    score = random.randint(10, 80)

    if ip not in user_history.get(user, {}).get("last_ips", []):
        score += 10
    if device not in user_history.get(user, {}).get("last_devices", []):
        score += 15
    if method not in user_history.get(user, {}).get("methods", []):
        score += 10

    context["risk_score"] = min(score, 100)
    return context

# --- 4. Device Trust Agent ---
def device_trust_agent(context: Dict) -> Dict:
    trusted_devices = user_history.get(context["username"], {}).get("last_devices", [])
    context["device_trusted"] = context["device_id"] in trusted_devices
    return context

# --- 5. Method Profile Agent ---
def method_profile_agent(context: Dict) -> Dict:
    usual_methods = user_history.get(context["username"], {}).get("methods", [])
    context["method_profiled"] = context["login_method"] in usual_methods
    return context

# --- 6. Reasoning Agent ---
def reasoner(context: Dict) -> Dict:
    score = context["risk_score"]
    device_ok = context["device_trusted"]
    method_ok = context["method_profiled"]
    explanation = ""

    if score > 85:
        action = "block"
        explanation = f"High risk score ({score})."
    elif not device_ok:
        action = "mfa"
        explanation = f"Unrecognized device."
    elif not method_ok:
        action = "mfa"
        explanation = f"Unusual login method: {context['login_method']}"
    elif score > 60:
        action = "mfa"
        explanation = f"Moderate risk score ({score}). MFA triggered."
    else:
        action = "allow"
        explanation = f"Login allowed. Low risk ({score}) with known device and method."

    context["decision"] = action
    context["explanation"] = explanation
    return context

# --- 7. History Update Agent ---
def update_history(context: Dict) -> Dict:
    if context["decision"] != "block":
        hist = user_history.setdefault(context["username"], {})
        hist["last_ips"] = list(set(hist.get("last_ips", []) + [context["ip_address"]]))
        hist["last_devices"] = list(set(hist.get("last_devices", []) + [context["device_id"]]))
        hist["last_login"] = context["timestamp"]
        hist["methods"] = list(set(hist.get("methods", []) + [context["login_method"]]))
    return context

# --- 8. LangGraph Workflow ---
workflow = StateGraph(input_schema=Dict)
workflow.add_node("IFMScorer", ToolNode(ifm_score))
workflow.add_node("DeviceTrust", ToolNode(device_trust_agent))
workflow.add_node("MethodProfile", ToolNode(method_profile_agent))
workflow.add_node("Reasoner", ToolNode(reasoner))
workflow.add_node("Updater", ToolNode(update_history))

workflow.set_entry_point("IFMScorer")
workflow.add_edge("IFMScorer", "DeviceTrust")
workflow.add_edge("DeviceTrust", "MethodProfile")
workflow.add_edge("MethodProfile", "Reasoner")
workflow.add_edge("Reasoner", "Updater")
workflow.set_finish_point("Updater")

graph_executor = workflow.compile()

# --- 9. Endpoint ---
@app.post("/agentic-login-netbanking", response_model=DecisionResponse)
def login_netbanking(req: LoginRequest):
    context = req.dict()
    result = graph_executor.invoke(context)
    return DecisionResponse(action=result["decision"], explanation=result["explanation"])

# --- 10. Run and Test ---
# uvicorn main:app --reload
# curl -X POST http://127.0.0.1:8000/agentic-login-netbanking \
# -H "Content-Type: application/json" \
# -d '{"username": "alice", "ip_address": "10.0.0.1", "device_id": "device_999", "login_method": "voice", "timestamp": "2025-06-12T12:00:00"}'