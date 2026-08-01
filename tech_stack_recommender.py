from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

DATASET_FILENAME = "raw_skills.csv"
MIN_REQUIRED_SKILLS = 3          
DEFAULT_TOP_N = 3                
DIVIDER = "=" * 62
SUB_DIVIDER = "-" * 62

TRENDING_FALLBACK_ROLES = [
    "Data Scientist",
    "Full Stack Developer",
    "DevOps Engineer",
]

@dataclass(frozen=True)
class JobRole:
    """Represents a single recommendable item (a job role) in the dataset."""

    name: str
    skills: Tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    """Represents one scored, ranked recommendation returned to the user."""

    role: JobRole
    score: float
    matched_skills: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def confidence_percent(self) -> float:
        """Cosine similarity (0..1) expressed as an intuitive percentage."""
        return round(self.score * 100, 1)

def normalize_token(raw_token: str) -> str:
    """Standardize a skill/tag string so vocabulary matching never fails
    due to whitespace or case differences (e.g. ' Python ' == 'python')."""
    return raw_token.strip().lower()


def load_job_roles(csv_path: str) -> List[JobRole]:
    """Load and validate the job-role dataset from a CSV file.

    Expected CSV columns: role, skills (comma-separated within the cell).
    Raises FileNotFoundError / ValueError with a clear message on failure
    so the CLI can handle it gracefully instead of crashing.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: '{csv_path}'")

    roles: List[JobRole] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "role" not in reader.fieldnames or "skills" not in reader.fieldnames:
            raise ValueError("Dataset CSV must contain 'role' and 'skills' columns.")

        for row_number, row in enumerate(reader, start=2):
            role_name = (row.get("role") or "").strip()
            skills_cell = (row.get("skills") or "").strip()

            if not role_name or not skills_cell:
        
                continue

            tokens = tuple(
                sorted(
                    {normalize_token(skill) for skill in skills_cell.split(",") if skill.strip()}
                )
            )
            if tokens:
                roles.append(JobRole(name=role_name, skills=tokens))

    if not roles:
        raise ValueError("Dataset contains no usable job roles.")

    return roles

class TfidfVectorizer:
    """A minimal, dependency-free TF-IDF vectorizer.

    Builds a shared vocabulary space from a corpus of "documents" (each
    document being a tuple of skill tokens), then converts any token
    collection - whether an item's skill list or a user's stated
    preferences - into a weighted numerical vector within that space.
    """

    def __init__(self, corpus: Sequence[Sequence[str]]):
        self._vocabulary: Tuple[str, ...] = tuple(
            sorted({token for document in corpus for token in document})
        )
        self._index_of: Dict[str, int] = {
            token: position for position, token in enumerate(self._vocabulary)
        }
        self._idf: Dict[str, float] = self._compute_idf(corpus)

    @property
    def vocabulary(self) -> Tuple[str, ...]:
        return self._vocabulary

    def _compute_idf(self, corpus: Sequence[Sequence[str]]) -> Dict[str, float]:
        """IDF = log(total_documents / documents_containing_term).

        The logarithm dampens the penalty for very common terms so scores
        stay well-behaved (per the briefing's "dampening effect" note).
        A +1 smoothing constant avoids division-by-zero for terms that,
        in edge cases, appear in every document.
        """
        total_documents = len(corpus)
        document_frequency: Dict[str, int] = {token: 0 for token in self._vocabulary}

        for document in corpus:
            for token in set(document):
                document_frequency[token] += 1

        return {
            token: math.log((total_documents + 1) / (document_frequency[token] + 1)) + 1.0
            for token in self._vocabulary
        }

    def _compute_tf(self, tokens: Sequence[str]) -> Dict[str, float]:
        """TF = (count of term in document) / (total terms in document)."""
        total_terms = len(tokens)
        if total_terms == 0:
            return {}

        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        return {token: count / total_terms for token, count in term_counts.items()}

    def transform(self, tokens: Sequence[str]) -> List[float]:
        """Convert a token collection into a dense TF-IDF weighted vector
        aligned to the shared vocabulary space (unknown tokens are simply
        ignored, matching the briefing's vector-mapping example)."""
        tf = self._compute_tf(tokens)
        vector = [0.0] * len(self._vocabulary)

        for token, tf_score in tf.items():
            position = self._index_of.get(token)
            if position is not None:
                vector[position] = tf_score * self._idf[token]

        return vector

def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """cos(theta) = (A . B) / (||A|| * ||B||)

    Returns a score in [0, 1] for our non-negative TF-IDF vectors, where
    1.0 means perfectly aligned interests and 0.0 means no overlap at all.
    Magnitude-invariant by design, which is why it is preferred over raw
    Euclidean distance for this use case (see briefing: "Why Euclidean
    Distance Fails at Scale").
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0  

    return dot_product / (magnitude_a * magnitude_b)

class TechStackRecommender:
    """Content-Based Filtering recommendation engine.

    Implements the 4-step pipeline from the briefing:
        1. Ingestion  - accept and clean the user's raw skill inputs
        2. Scoring    - TF-IDF vectorize + cosine-score against every role
        3. Sorting    - rank roles by descending similarity score
        4. Filtering  - truncate to a Top-N list to prevent choice overload
    """

    def __init__(self, roles: List[JobRole]):
        if not roles:
            raise ValueError("Recommender requires at least one job role.")

        self._roles = roles
        self._vectorizer = TfidfVectorizer([role.skills for role in roles])
        self._role_vectors: List[List[float]] = [
            self._vectorizer.transform(role.skills) for role in roles
        ]

    @property
    def known_skills(self) -> Tuple[str, ...]:
        return self._vectorizer.vocabulary

    @property
    def roles(self) -> Tuple[JobRole, ...]:
        return tuple(self._roles)

    def recommend(
        self, user_skills: Sequence[str], top_n: int = DEFAULT_TOP_N
    ) -> List[Recommendation]:
        """Run the full Ingestion -> Scoring -> Sorting -> Filtering pipeline
        and return the Top-N ranked recommendations for the given skills."""
        cleaned_skills = tuple(sorted({normalize_token(s) for s in user_skills if s.strip()}))
        user_vector = self._vectorizer.transform(cleaned_skills)
        scored: List[Recommendation] = []
        for role, role_vector in zip(self._roles, self._role_vectors):
            score = cosine_similarity(user_vector, role_vector)
            matched = tuple(skill for skill in cleaned_skills if skill in role.skills)
            scored.append(Recommendation(role=role, score=score, matched_skills=matched))
        scored.sort(key=lambda rec: (-rec.score, -len(rec.matched_skills), rec.role.name))
        return scored[: max(top_n, 1)]

    def trending_fallback(self, top_n: int = DEFAULT_TOP_N) -> List[Recommendation]:
        """Cold-start bypass: return globally popular/trending roles with a
        neutral (unscored) recommendation when a user profile shares no
        vocabulary overlap with the dataset at all."""
        lookup = {role.name: role for role in self._roles}
        fallback_roles = [lookup[name] for name in TRENDING_FALLBACK_ROLES if name in lookup]
        if not fallback_roles:
            fallback_roles = self._roles[:top_n]

        return [
            Recommendation(role=role, score=0.0, matched_skills=())
            for role in fallback_roles[:top_n]
        ]

def parse_skill_input(raw_input: str) -> List[str]:
    """Parse a comma-separated raw string into a clean, de-duplicated list
    of skill tokens. Handles extra whitespace, mixed case, empty entries,
    and duplicate entries gracefully."""
    if not raw_input:
        return []

    seen: Dict[str, None] = {}
    for chunk in raw_input.split(","):
        token = normalize_token(chunk)
        if token:
            seen.setdefault(token, None)  
    return list(seen.keys())

def prompt_for_skills(minimum_required: int = MIN_REQUIRED_SKILLS) -> List[str]:
    """Repeatedly prompt the user until a valid set of skills is provided.
    Never crashes on empty input, duplicates, casing, or stray whitespace."""
    while True:
        print(f"\nEnter at least {minimum_required} skills or interests, separated by commas.")
        print("  Example: python, cloud computing, automation")
        raw_input_value = input("Your skills > ")

        skills = parse_skill_input(raw_input_value)

        if len(skills) < minimum_required:
            print(
                f"  [!] Please provide at least {minimum_required} distinct skills "
                f"(you entered {len(skills)}). Let's try again."
            )
            continue

        return skills


def prompt_for_top_n(default: int = DEFAULT_TOP_N, maximum: int = 10) -> int:
    """Ask the user how many recommendations they'd like, with a safe
    default and full validation against non-numeric or out-of-range input."""
    raw_value = input(f"\nHow many recommendations would you like? [default: {default}] > ").strip()

    if not raw_value:
        return default

    if not raw_value.isdigit():
        print(f"  [!] '{raw_value}' isn't a valid number. Using default of {default}.")
        return default

    value = int(raw_value)
    if value < 1:
        print(f"  [!] Must be at least 1. Using default of {default}.")
        return default
    if value > maximum:
        print(f"  [!] Capping at the maximum of {maximum} to avoid choice overload.")
        return maximum

    return value


def prompt_menu_choice(valid_choices: Sequence[str]) -> str:
    """Read and validate a menu choice, tolerant of case and whitespace."""
    while True:
        choice = input("\nSelect an option > ").strip().lower()
        if choice in valid_choices:
            return choice
        print(f"  [!] '{choice}' is not a valid option. Please choose one of: {', '.join(valid_choices)}.")

def print_header(title: str) -> None:
    print(f"\n{DIVIDER}\n{title.center(62)}\n{DIVIDER}")


def print_welcome_screen() -> None:
    print(DIVIDER)
    print("DECODELABS  |  ARTIFICIAL INTELLIGENCE INTERNSHIP".center(62))
    print("PROJECT 3 : TECH STACK RECOMMENDER".center(62))
    print(DIVIDER)
    print(
        "\nWelcome, AI Engineer.\n"
        "This tool maps your raw skills and career goals to the job\n"
        "roles that align with them, using Content-Based Filtering\n"
        "powered by TF-IDF feature weighting and Cosine Similarity."
    )


def print_main_menu() -> None:
    print_header("MAIN MENU")
    print("  1. Get personalized job role recommendations")
    print("  2. Browse all available job roles")
    print("  3. View the known skill vocabulary")
    print("  4. Exit")


def print_recommendations(recommendations: List[Recommendation], is_fallback: bool = False) -> None:
    print_header("TRENDING ROLES (COLD START)" if is_fallback else "RECOMMENDED JOB ROLES")

    if is_fallback:
        print(
            "None of your skills matched our known vocabulary, so here are\n"
            "our currently trending roles to help you get started:\n"
        )

    if not recommendations:
        print("No recommendations could be generated.")
        return

    for rank, rec in enumerate(recommendations, start=1):
        print(f"\n#{rank}  {rec.role.name}")
        if not is_fallback:
            print(f"     Match confidence : {rec.confidence_percent}%")
            matched = ", ".join(rec.matched_skills) if rec.matched_skills else "none (matched by related concepts)"
            print(f"     Skills matched    : {matched}")
        print(f"     Full skill set    : {', '.join(rec.role.skills)}")
    print(f"\n{SUB_DIVIDER}")


def print_all_roles(roles: Sequence[JobRole]) -> None:
    print_header("AVAILABLE JOB ROLES")
    for index, role in enumerate(roles, start=1):
        print(f"\n{index:>2}. {role.name}")
        print(f"     Skills: {', '.join(role.skills)}")
    print(f"\n{SUB_DIVIDER}")


def print_vocabulary(vocabulary: Sequence[str]) -> None:
    print_header("KNOWN SKILL VOCABULARY")
    columns = 4
    for i in range(0, len(vocabulary), columns):
        row = vocabulary[i : i + columns]
        print("  " + "  |  ".join(skill.ljust(20) for skill in row))
    print(f"\n{SUB_DIVIDER}")


def print_goodbye() -> None:
    print_header("SESSION ENDED")
    print(
        "Thank you for using the DecodeLabs Tech Stack Recommender.\n"
        "Keep mapping your preferences, keep refining your logic -\n"
        "that's how real AI engineers are built.\n\n"
        "Goodbye, and happy building!"
    )
    print(DIVIDER)

def resolve_dataset_path() -> str:
    """Locate raw_skills.csv relative to this script so the app runs
    correctly regardless of the current working directory."""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_directory, DATASET_FILENAME)


def handle_recommend_flow(engine: TechStackRecommender) -> None:
    user_skills = prompt_for_skills()
    top_n = prompt_for_top_n()

    recommendations = engine.recommend(user_skills, top_n=top_n)
    best_score = recommendations[0].score if recommendations else 0.0

    if best_score <= 0.0:
        fallback = engine.trending_fallback(top_n=top_n)
        print_recommendations(fallback, is_fallback=True)
    else:
        print_recommendations(recommendations)


def run_application() -> None:
    dataset_path = resolve_dataset_path()

    try:
        roles = load_job_roles(dataset_path)
        engine = TechStackRecommender(roles)
    except (FileNotFoundError, ValueError) as error:
        print(f"\n[FATAL] Could not start the recommender: {error}")
        sys.exit(1)

    print_welcome_screen()

    menu_actions = {
        "1": lambda: handle_recommend_flow(engine),
        "2": lambda: print_all_roles(engine.roles),
        "3": lambda: print_vocabulary(engine.known_skills),
    }

    while True:
        print_main_menu()
        choice = prompt_menu_choice(valid_choices=("1", "2", "3", "4"))

        if choice == "4":
            print_goodbye()
            break

        menu_actions[choice]()


if __name__ == "__main__":
    try:
        run_application()
    except KeyboardInterrupt:
        print("\n\n[!] Session interrupted by user.")
        print_goodbye()
        sys.exit(0)
