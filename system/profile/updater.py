"""
Responsible for extracting user information from conversations and updating the user profile
automatically using the LLM.
"""

import json
from system.llm.client import LLMClient
from system.profile.schema import UserProfile


class ProfileUpdater:
    """
    Uses the LLM to extract structured user information from conversations and merges it into
    the existing user profile.

    Attributes:
        _llm: Private LLMClient instance for extraction requests.
    """

    def __init__(self) -> None:
        """
        Initializes the ProfileUpdater with an LLMClient instance.
        """
        self._llm = LLMClient()

    def extract_and_update(
        self, user_message: str, profile: UserProfile, username: str
    ) -> UserProfile:
        """
        Extracts information from a message and updates the profile.
        If extraction fails for any reason, returns the profile unchanged
        so the chat flow is never interrupted.

        Args:
            user_message: The current message from the user.
            profile: The current UserProfile instance to update.
            username: The username used to identify the profile file.

        Returns:
            An updated UserProfile instance, or the original one on failure.
        """
        extraction_prompt = f"""
You are an information extraction assistant.
Analyze the user message below and extract any personal information mentioned. Return ONLY
a valid JSON object with these exact keys:
- "name": string or null
- "interests": list of strings or null
- "goals": list of strings or null
- "values": list of strings or null

Rules:
- Only extract information explicitly mentioned in the message.
- If nothing relevant is found for a field, use null.
- Never invent or assume information.
- Return ONLY the JSON object, no explanation, no markdown.

User message: "{user_message}"
"""

        try:
            raw = self._llm.generate_response(
                extraction_prompt,
                model="openai/gpt-oss-20b",
                max_new_tokens=256,
            )
        except Exception as e:
            # Extraction is a best-effort side task: never let it break the chat.
            print(f"[ProfileUpdater] Extraction call failed, skipping update: {e}")
            return profile

        try:
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            extracted = json.loads(cleaned)
        except json.JSONDecodeError:
            return profile

        # Merge extracted info into existing profile
        if extracted.get("name"):
            profile.name = extracted["name"]
        if extracted.get("interests"):
            profile.interests = list(set(profile.interests + extracted["interests"]))
        if extracted.get("goals"):
            profile.goals = list(set(profile.goals + extracted["goals"]))
        if extracted.get("values"):
            profile.values = list(set(profile.values + extracted["values"]))

        profile.save(username)
        return profile
