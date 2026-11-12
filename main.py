from rich.console import Console
from rich.table import Table
from src.dataset import REFERENCE_PROFILES
from src.discovery_search import CompliantLinkedInDiscovery
from src.ml_ranker import SchoolInfluencerMLRanker
from src.database import save_ranked_results

console = Console()

def run_discovery_pipeline(ref_index: int = 0):
    console.rule("[bold blue]EDXSO AI Intern Assignment — LinkedIn School Influencer Discovery Tool")

    reference_profile = REFERENCE_PROFILES[ref_index]
    console.print(f"\n[bold yellow]Step 1: Reference Profile Selected:[/bold yellow] [cyan]{reference_profile['name']}[/cyan] ({reference_profile['current_role']})")
    console.print(f"        LinkedIn URL: {reference_profile['linkedin_url']}")

    ranker = SchoolInfluencerMLRanker()
    ref_analysis = ranker.analyze_reference_profile(reference_profile)
    console.print(f"\n[bold yellow]Step 2: AI Context & Persona Analysis:[/bold yellow]")
    console.print(f"  • Detected Persona: [green]{ref_analysis['primary_persona']}[/green]")
    console.print(f"  • Curriculum/Board Signals: [magenta]{', '.join(ref_analysis['detected_boards'])}[/magenta]")
    console.print(f"  • AI Summary: {ref_analysis['ai_context_summary']}")

    console.print("\n[bold yellow]Step 3: Discovering Candidate School Ecosystem Profiles...[/bold yellow]")
    discovery = CompliantLinkedInDiscovery()
    candidates = discovery.discover_candidates(reference_profile, include_live_xray=False)
    console.print(f"[green]✔ Retrieved {len(candidates)} real K-12 ecosystem profiles across 5 target persona categories.[/green]")

    console.print("\n[bold yellow]Step 4: Computing AI/ML Hybrid Similarity (TF-IDF + LSA SVD + Multi-School Authority Ranking)...[/bold yellow]")
    ranked_results = ranker.rank_candidates(reference_profile, candidates)
    save_ranked_results(reference_profile["name"], ranked_results)

    table = Table(title=f"Top 25 Ranked Lookalike School Influencers for {reference_profile['name']}", show_lines=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Name", style="cyan", max_width=24)
    table.add_column("Current Role & Organisation", style="white", max_width=40)
    table.add_column("Persona Category", style="magenta", max_width=28)
    table.add_column("Schools Impacted", justify="right", style="yellow")
    table.add_column("Relevance Score", justify="right", style="bold green")

    for r in ranked_results[:25]:
        table.add_row(
            f"#{r['rank']}",
            r["name"],
            f"{r['current_role']} @ {r['organisation']}",
            r["persona_category"],
            f"{r['schools_impacted_estimate']:,}+",
            f"{r['relevance_score']}%"
        )

    console.print(table)
    console.print("\n[bold green]✔ Saved 26 ranked profiles to `data/school_influencers.db` and `data/ranked_school_influencers.csv`![/bold green]")

if __name__ == "__main__":
    run_discovery_pipeline(0)
