import asyncio
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# This is the key. We are importing the powerful, pre-built client.
from llm_interface.tennis_api_client import ProfessionalTennisAPIClient

# --- CONFIGURATION ---

# 1. Add all the player names you need to analyze here.
PLAYERS_TO_ANALYZE = [
    "Carlos Alcaraz",
    "Jannik Sinner",
    "Novak Djokovic",
    "Jack Draper",
    "Alexander Zverev",
    "Lorenzo Musetti",
    "Tommy Paul",
    "Casper Ruud",
    "Daniil Medvedev",
    "Alex De Minaur",
    "Arthur Fils",
    "Taylor Fritz",
    "Grigor Dimitrov",
    "Holger Rune",
    "Matteo Berrettini",
    "Stefanos Tsitsipas",
    "Hubert Hurkacz",
    "Francisco Cerundolo",
    "Gael Monfils",
    "Joao Fonseca",
    "Tallon Griekspoor",
    "Ben Shelton",
    "Jiri Lehecka",
    "Tomas Machac",
    "Jakub Mensik",
    "Andrey Rublev",
    "Karen Khachanov",
    "Ugo Humbert",
    "Alejandro Davidovich Fokina",
    "Sebastian Korda",
    "Thanasi Kokkinakis",
    "Jordan Thompson",
    "Frances Tiafoe",
    "Matteo Arnaldi",
    "Denis Shapovalov",
    "Flavio Cobolli",
    "Alexei Popyrin",
    "Felix Auger Aliassime",
    "Alexander Bublik",
    "Jacob Fearnley",
    "Cameron Norrie",
    "Marcos Giron",
    "Fabian Marozsan",
    "Nuno Borges",
    "Laslo Djere",
    "Alex Michelsen",
    "Jenson Brooksby",
    "Gabriel Diallo",
    "Giovanni Mpetshi Perricard",
    "Colton Smith",
    "Hamad Medjedovic",
    "Brandon Nakashima",
    "Ethan Quinn",
    "Borna Coric",
    "Zizou Bergs",
    "Alexandre Muller",
    "Roberto Bautista Agut",
    "David Goffin",
    "Kei Nishikori",
    "Juncheng Shang",
    "Marin Cilic",
    "Marton Fucsovics",
    "Miomir Kecmanovic",
    "Roman Safiullin",
    "Mariano Navone",
    "Corentin Moutet",
    "Yoshihito Nishioka",
    "Quentin Halys",
    "Reilly Opelka",
    "Sebastian Ofner",
    "Tomas Martin Etcheverry",
    "Daniel Altmaier",
    "Lucas Pouille",
    "Jaume Munar",
    "Alejandro Tabilo",
    "Lorenzo Sonego",
    "Filip Misolic",
    "Luciano Darderi",
    "Camilo Ugo Carabelli",
    "Learner Tien",
    "Jason Kubler",
    "Kamil Majchrzak",
    "Pablo Carreno Busta",
    "Roberto Carballes Baena",
    "Yibing Wu",
    "Vit Kopriva",
    "Borna Gojo",
    "Zhizhen Zhang",
    "Christopher Oconnell",
    "Stan Wawrinka",
    "Cristian Garin",
    "Dominik Koepfer",
    "Raphael Collignon",
    "Benjamin Bonzi",
    "Jan Lennard Struff",
    "Nicolas Jarry",
    "Sebastian Baez",
    "Bu Yunchaokete",
    "Jesper De Jong",
    "Francisco Comesana",
]

# 2. Where to save the final .txt documents for your RAG app.
#    This will be created on your Desktop for easy access.
OUTPUT_DOCS_FOLDER = Path.home() / "Desktop" / "TENNIS_KNOWLEDGE_BASE_API"


# --- END CONFIGURATION ---

def format_player_data_for_rag(player_name: str, data: dict) -> str:
    """
    Takes the complex JSON data from the API and formats it into a clean,
    human-readable text document for your RAG to ingest.
    """
    if not data or data.get("error"):
        return f"Could not retrieve detailed analysis for {player_name}."

    profile = data.get("profile", {})
    details = data.get("player_details", {}).get("player", {})
    form = data.get("recent_form_string", "N/A")

    # Start building the text document
    content = f"Comprehensive Player Analysis for: {profile.get('name', player_name)}\n\n"
    content += f"## Player Profile\n"
    content += f"- Name: {profile.get('name', 'N/A')}\n"
    content += f"- Country: {details.get('country', {}).get('name', 'N/A')}\n"
    content += f"- Current Ranking: {profile.get('ranking', 'N/A')}\n"
    content += f"- Plays: {details.get('plays', 'N/A')}\n"
    content += f"- Turned Pro: {details.get('turnedPro', 'N/A')}\n\n"

    content += f"## Recent Form\n"
    content += f"- Last 10 Matches (W/L): {form}\n\n"

    # Add recent events if available
    recent_events = data.get("recent_events", {}).get("events", [])
    if recent_events:
        content += f"## Recent Tournament Results (last {len(recent_events)})\n"
        for event in recent_events[:5]:  # Show last 5 for brevity
            tournament = event.get('tournament', {}).get('name', 'Unknown')
            opponent = event.get('homeTeam' if event.get('awayTeam', {}).get('id') == profile.get('id') else 'awayTeam',
                                 {}).get('name', 'Unknown')
            score = f"{event.get('homeScore', {}).get('display', 0)}-{event.get('awayScore', {}).get('display', 0)}"
            winner_code = event.get('winnerCode')
            result = "Win" if (event.get('homeTeam', {}).get('id') == profile.get('id') and winner_code == 1) or \
                              (event.get('awayTeam', {}).get('id') == profile.get(
                                  'id') and winner_code == 2) else "Loss"
            content += f"- {tournament}: {result} vs {opponent} ({score})\n"
        content += "\n"

    # Add ranking history if available
    rankings_history = data.get("rankings_history", {}).get("rankings", [])
    if rankings_history:
        content += f"## Ranking History\n"
        for rank_entry in rankings_history[:3]:  # Show last 3 ranking entries
            date = rank_entry.get("updatedAtTimestamp")
            rank = rank_entry.get("ranking")
            points = rank_entry.get("points")
            content += f"- Date: {date}, Rank: {rank}, Points: {points}\n"
        content += "\n"

    return content.strip()


async def main():
    """
    Main function to run the API scraper.
    """
    print("--- 🚀 Starting Automated Tennis API Scraper ---")

    # Load the TENNIS_RAPIDAPI_KEY from your .env file
    load_dotenv()
    api_key = os.getenv("TENNIS_RAPIDAPI_KEY")
    if not api_key:
        print("❌ CRITICAL ERROR: TENNIS_RAPIDAPI_KEY not found in .env file.")
        print("   Please make sure your .env file is in the project root and contains the key.")
        return

    print(f"✅ Found API Key. Analyzing {len(PLAYERS_TO_ANALYZE)} players...")

    # Create the output directory and any necessary parent directories
    OUTPUT_DOCS_FOLDER.mkdir(parents=True, exist_ok=True)  # <<< THIS IS THE FIX
    print(f"✅ Output folder is ready at: {OUTPUT_DOCS_FOLDER.resolve()}")

    # Initialize the API client
    client = ProfessionalTennisAPIClient()

    try:
        for player_name in PLAYERS_TO_ANALYZE:
            print(f"\nFetching data for {player_name}...")
            # This one function call gets all the data we need for a player
            analysis_data = await client.get_comprehensive_player_analysis(player_name)

            if analysis_data.get("error"):
                print(f"  ❌ Error fetching data for {player_name}: {analysis_data['error']}")
                continue

            # Format the data into a text document
            rag_document_text = format_player_data_for_rag(player_name, analysis_data)

            # Save the document
            filename = player_name.lower().replace(" ", "_") + "_analysis.txt"
            output_path = OUTPUT_DOCS_FOLDER / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rag_document_text)

            print(f"  ✅ Successfully created document: {filename}")

            # Be a good citizen and pause between players
            await asyncio.sleep(1)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        await client.close()
        print("\n--- ✅ API Client closed. ---")

    print(f"\n🎉🎉🎉 All Done! Your new knowledge base documents are in '{OUTPUT_DOCS_FOLDER}'")


if __name__ == "__main__":
    asyncio.run(main())