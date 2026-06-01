# memory.py

class SessionMemory:

    def __init__(self):
        self.history = []

    def add_turn(
        self,
        user_message,
        assistant_message
    ):

        self.history.append(
            {
                "user": user_message,
                "assistant": assistant_message
            }
        )

        # keep last 10 turns
        self.history = self.history[-10:]

    def get_history(self):

        if not self.history:
            return ""

        text = []

        for turn in self.history:

            text.append(
                f"User: {turn['user']}"
            )

            text.append(
                f"Assistant: {turn['assistant']}"
            )

        return "\n".join(text)