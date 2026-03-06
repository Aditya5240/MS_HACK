import os
import datetime
from dotenv import load_dotenv

# Google API Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# LiveKit Imports
from livekit import agents, rtc
from livekit.agents import (
    AgentServer, 
    AgentSession, 
    Agent, 
    room_io, 
    function_tool, 
    RunContext, 
    ToolError
)
from livekit.plugins import silero, groq, deepgram, cartesia, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

class Assistant(Agent): 
    def __init__(self):
        super().__init__(
            instructions="""You are a professional voice assistant. 
            Today is Friday, March 6, 2026. 
            When a user provides event details, call the 'create_calendar_event' tool EXACTLY ONCE.
            Once you receive a 'FINAL_CONFIRMATION' from the tool, simply tell the user it is done.
            DO NOT attempt to call the tool again for the same request.""",
        )
        # Track events created in this session to prevent duplication
        self._created_events = set()

    @function_tool
    def create_calendar_event(
        self,
        ctx: RunContext,
        title: str,
        start_time: str,
        end_time: str = "",
        description: str = ""
    ) -> str:
        """
        Create a Google Calendar event.
        Args:
            title: The name of the event.
            start_time: ISO 8601 string (e.g., '2026-03-09T22:00:00').
            end_time: ISO 8601 string.
        """
        # 1. State Check: Prevent duplicate execution in the same turn
        event_key = f"{title}-{start_time}"
        if event_key in self._created_events:
            print(f"DEBUG: Blocked duplicate attempt for {event_key}")
            return "FINAL_CONFIRMATION: This event has already been successfully created. Do not call this tool again."

        try:
            print(f"DEBUG: Creating Calendar Event: {title}")
            service = get_calendar_service()
            
            if not end_time or end_time == "" or end_time == start_time:
                dt = datetime.datetime.fromisoformat(start_time.replace('Z', ''))
                end_time = (dt + datetime.timedelta(hours=1)).isoformat()

            event = {
                'summary': title,
                'description': description,
                'start': {'dateTime': start_time, 'timeZone': 'Asia/Kolkata'},
                'end': {'dateTime': end_time, 'timeZone': 'Asia/Kolkata'},
            }

            service.events().insert(calendarId='primary', body=event).execute()
            
            # Record that we successfully created this
            self._created_events.add(event_key)
            
            # Returning a very explicit string helps the LLM understand the task is finished
            return f"FINAL_CONFIRMATION: The event '{title}' is now on the calendar for {start_time}."
        
        except Exception as e:
            raise ToolError(f"Google Calendar Error: {str(e)}")

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
    # Retrieve time for the greeting
    now = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
    
    agent_instance = Assistant()

    session = AgentSession(
        stt=deepgram.STT(),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=cartesia.TTS(voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(room=ctx.room, agent=agent_instance)

    await session.generate_reply(
        instructions=f"Current time is {now}. Greet the user and offer to manage their calendar."
    )

if __name__ == "__main__":
    agents.cli.run_app(server)