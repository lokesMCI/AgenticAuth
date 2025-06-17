import asyncio
from langchain.agents import create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from typing import Any, Dict
import json

# === Define MCP Tools with async support ===
class GetGeolocationTool(BaseTool):
    name = "get_geolocation"
    description = "Retrieve user geolocation data based on IP or device context."

    def _run(self, query: str) -> str:
        return "US, California"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetIPTool(BaseTool):
    name = "get_ip"
    description = "Retrieve user IP address."

    def _run(self, query: str) -> str:
        return "192.168.0.1"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetTypingSpeedTool(BaseTool):
    name = "get_typing_speed"
    description = "Measure user's typing speed and pattern."

    def _run(self, query: str) -> str:
        return "45 wpm"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class AskPersonalQATool(BaseTool):
    name = "ask_personal_qa"
    description = "Ask user personal challenge questions for authentication."

    def _run(self, query: str) -> str:
        return "Mother's maiden name?"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetDeviceFingerprintTool(BaseTool):
    name = "get_device_fingerprint"
    description = "Fetch browser and device fingerprint (OS, browser, resolution, fonts, etc.)."

    def _run(self, query: str) -> str:
        return "FingerprintHash: abc123"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetBiometricFaceTool(BaseTool):
    name = "get_face_biometric"
    description = "Verify face biometric using camera."

    def _run(self, query: str) -> str:
        return "Face match score: 0.96"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetBiometricFingerprintTool(BaseTool):
    name = "get_fingerprint_biometric"
    description = "Verify fingerprint biometric via hardware device."

    def _run(self, query: str) -> str:
        return "Fingerprint match score: 0.94"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetVoiceBiometricTool(BaseTool):
    name = "get_voice_biometric"
    description = "Verify voice biometric."

    def _run(self, query: str) -> str:
        return "Voice match score: 0.91"
    async def _arun(self, query: str) -> str:
        return self._run(query)

class SendSMSTokenTool(BaseTool):
    name = "send_sms_otp"
    description = "Send an SMS OTP for verification."

    def _run(self, query: str) -> str:
        return "OTP sent via SMS."
    async def _arun(self, query: str) -> str:
        return self._run(query)

class SendEmailTokenTool(BaseTool):
    name = "send_email_otp"
    description = "Send an Email OTP for verification."

    def _run(self, query: str) -> str:
        return "OTP sent via Email."
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetAuthenticatorTokenTool(BaseTool):
    name = "get_authenticator_otp"
    description = "Get OTP from user's authenticator app (e.g. Google Authenticator, Authy)."

    def _run(self, query: str) -> str:
        return "Authenticator OTP accepted."
    async def _arun(self, query: str) -> str:
        return self._run(query)

class GetHardwareTokenTool(BaseTool):
    name = "get_hardware_token"
    description = "Request OTP from hardware token (e.g. RSA, YubiKey)."

    def _run(self, query: str) -> str:
        return "Hardware token accepted."
    async def _arun(self, query: str) -> str:
        return self._run(query)

class PushNotificationTool(BaseTool):
    name = "send_push_notification"
    description = "Send a push notification for approval to the user’s trusted device."

    def _run(self, query: str) -> str:
        return "Push notification sent."
    async def _arun(self, query: str) -> str:
        return self._run(query)

class RiskAnalysisTool(BaseTool):
    name = "get_risk_analysis"
    description = "Return a summary risk score based on user behavior, device, and location."

    def _run(self, query: str) -> str:
        return "Risk score: 0.72"
    async def _arun(self, query: str) -> str:
        return self._run(query)

# === Policy Document System Prompt ===
policy = '''
You are the primary authentication agent for a digital banking platform. Your role is to intelligently decide, based on the selected feature and risk policy, which minimal combination of authentication methods (tools) to trigger for verifying the user. You MUST adhere to risk-based authentication principles.

Authentication methods available (via tools):
- Device & Browser Fingerprinting
- IP & Geolocation
- Typing behavior
- Personal Q&A challenge
- Biometric: Face / Fingerprint / Voice
- OTP: SMS / Email / Authenticator App / Hardware token
- Push-based Approval
- Contextual Risk Analysis

Risk levels and example features:
Low Risk (password + passive methods like fingerprinting, typing, IP):
- View dashboard, e-statements, profile

Medium Risk (add OTP or challenge question):
- Fund transfers (IMPS, NEFT, internal), bill payments, standing instructions

Medium–High (3-D secure or soft OTP/push + context analysis):
- Online commerce, recovery/password reset

High Risk (multi-factor: OTP + biometric/device + risk score):
- KYC update, nominee changes, large/third-party transfers, bank detail changes

Behavior:
- NEVER trigger all tools blindly.
- First decide the risk tier from the feature.
- Try passive verification first: location, device, typing.
- Escalate step-by-step using only what’s needed to cross the risk threshold.
- If confident threshold is met, return observations.
- If not enough, request additional info (specific tools).
- Prioritize user convenience without compromising on security.
'''

# === Instantiate Chat Model ===
llm = ChatOpenAI(temperature=0)

# === React Agent: decides which tools to invoke ===
auth_agent = create_react_agent(
    llm=llm,
    tools=[
        GetGeolocationTool(),
        GetIPTool(),
        GetTypingSpeedTool(),
        AskPersonalQATool(),
        GetDeviceFingerprintTool(),
        GetBiometricFaceTool(),
        GetBiometricFingerprintTool(),
        GetVoiceBiometricTool(),
        SendSMSTokenTool(),
        SendEmailTokenTool(),
        GetAuthenticatorTokenTool(),
        GetHardwareTokenTool(),
        PushNotificationTool(),
        RiskAnalysisTool(),
    ],
    system_message=policy,
)

# === Decider Agent: evaluates collected data against IMF risk threshold ===
class DeciderAgent:
    def __init__(self, llm_model: ChatOpenAI, risk_threshold: float = 0.7):
        self.llm = llm_model
        self.threshold = risk_threshold

    async def evaluate(self, feature: str, context: str) -> Dict[str, Any]:
        prompt = (
            f"Feature: {feature}\n"
            f"Collected data: {context}\n"
            "Assess if authentication context meets risk threshold. "
            "Return JSON with keys: 'decision'('proceed'|'more_info'|'deny'), 'risk_score'(0-1), and 'missing_info'(list)."
        )
        resp = await self.llm.arun(prompt)
        return json.loads(resp)

# === Orchestration Loop ===
async def authenticate(feature_name: str):
    max_rounds = 3
    context = ""
    decider = DeciderAgent(llm)

    for round_idx in range(1, max_rounds + 1):
        # Step 1: React agent gathers data
        prompt = f"User clicked '{feature_name}'. Collect required factors and output key:value pairs."  
        observations = await auth_agent.arun(prompt)
        context += observations + "\n"

        # Step 2: Decider evaluates
        decision = await decider.evaluate(feature_name, context)
        decision_type = decision.get("decision")

        if decision_type == "proceed":
            print(f"Authentication successful for '{feature_name}' (risk_score={decision['risk_score']})")
            return True
        elif decision_type == "deny":
            print(f"Authentication denied for '{feature_name}' (risk_score={decision['risk_score']})")
            return False
        else:
            missing = decision.get("missing_info", [])
            print(f"Round {round_idx}: missing info {missing}, requesting additional factors...")
            context += "Missing:" + ",".join(missing) + "\n"

    print(f"Authentication failed after {max_rounds} rounds for '{feature_name}'.")
    return False

# === Entry Point ===
async def main():
    features = [
        "View account info",
        "Fund transfers (IMPS)",
        "Large transfers to third-party",
        "Credential recovery/password reset"
    ]
    for feat in features:
        result = await authenticate(feat)
        print(f"Final result for {feat}: {result}\n")

if __name__ == "__main__":
    asyncio.run(main())
