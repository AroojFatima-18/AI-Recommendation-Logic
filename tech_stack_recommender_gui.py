from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import tkinter as tk
from tkinter import ttk, messagebox
DATASET_FILENAME = "raw_skills.csv"
MIN_REQUIRED_SKILLS = 3
DEFAULT_TOP_N = 3
MAX_TOP_N = 10

TRENDING_FALLBACK_ROLES = [
    "Data Scientist",
    "Full Stack Developer",
    "DevOps Engineer",
]

EMBEDDED_DATASET: List[Tuple[str, str]] = [
    ("Data Scientist", "python,sql,machine learning,statistics,data analysis,pandas,numpy,data visualization"),
    ("DevOps Engineer", "aws,docker,kubernetes,ci/cd,automation,linux,terraform,cloud computing"),
    ("Backend Developer", "java,python,sql,apis,microservices,git,databases,rest"),
    ("Frontend Developer", "javascript,html,css,react,web design,ui/ux,typescript,responsive design"),
    ("Full Stack Developer", "javascript,python,react,node.js,sql,html,css,apis,git"),
    ("Cloud Architect", "aws,azure,cloud computing,networking,security,terraform,automation,kubernetes"),
    ("Machine Learning Engineer", "python,machine learning,tensorflow,pytorch,data structures,algorithms,sql,statistics"),
    ("Data Engineer", "python,sql,etl,spark,data pipelines,cloud computing,aws,data structures"),
    ("Cybersecurity Analyst", "security,networking,linux,penetration testing,risk assessment,firewalls,cryptography"),
    ("Mobile App Developer", "java,kotlin,swift,mobile development,ui/ux,git,apis,android"),
    ("QA Engineer", "testing,automation,selenium,python,java,bug tracking,ci/cd,quality assurance"),
    ("Systems Administrator", "linux,networking,scripting,automation,windows server,security,troubleshooting"),
    ("Database Administrator", "sql,database design,performance tuning,backup recovery,oracle,mysql,postgresql"),
    ("Site Reliability Engineer", "linux,automation,monitoring,ci/cd,kubernetes,cloud computing,scripting,incident response"),
    ("Business Intelligence Analyst", "sql,data visualization,excel,power bi,tableau,data analysis,reporting"),
    ("Software Architect", "system design,java,python,microservices,scalability,design patterns,cloud computing"),
    ("Network Engineer", "networking,security,routing,switching,firewalls,troubleshooting,linux"),
    ("Game Developer", "c++,unity,game design,graphics programming,algorithms,data structures,git"),
    ("UI/UX Designer", "ui/ux,web design,figma,prototyping,user research,html,css,responsive design"),
    ("Blockchain Developer", "blockchain,solidity,cryptography,smart contracts,python,security,git"),
]
@dataclass(frozen=True)
class JobRole:
    name: str
    skills: Tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    role: JobRole
    score: float
    matched_skills: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def confidence_percent(self) -> float:
        return round(self.score * 100, 1)
def normalize_token(raw_token: str) -> str:
    return raw_token.strip().lower()


def parse_skill_input(raw_input: str) -> List[str]:
    """Comma-separated string -> clean, de-duplicated, order-preserving list.
    Tolerant of empty entries, stray whitespace, and mixed casing."""
    if not raw_input:
        return []
    seen: Dict[str, None] = {}
    for chunk in raw_input.split(","):
        token = normalize_token(chunk)
        if token:
            seen.setdefault(token, None)
    return list(seen.keys())


def resolve_dataset_path() -> str:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_directory, DATASET_FILENAME)


def load_job_roles() -> List[JobRole]:
    """Load roles from raw_skills.csv next to this script if present,
    otherwise fall back to the embedded dataset so the app never fails
    to launch just because a data file is missing."""
    csv_path = resolve_dataset_path()
    rows: List[Tuple[str, str]] = []

    if os.path.isfile(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and "role" in reader.fieldnames and "skills" in reader.fieldnames:
                    for row in reader:
                        role_name = (row.get("role") or "").strip()
                        skills_cell = (row.get("skills") or "").strip()
                        if role_name and skills_cell:
                            rows.append((role_name, skills_cell))
        except (OSError, csv.Error):
            rows = []

    if not rows:
        rows = EMBEDDED_DATASET

    roles: List[JobRole] = []
    for role_name, skills_cell in rows:
        tokens = tuple(sorted({normalize_token(s) for s in skills_cell.split(",") if s.strip()}))
        if tokens:
            roles.append(JobRole(name=role_name, skills=tokens))
    return roles

class TfidfVectorizer:
    def __init__(self, corpus: Sequence[Sequence[str]]):
        self._vocabulary: Tuple[str, ...] = tuple(sorted({t for doc in corpus for t in doc}))
        self._index_of: Dict[str, int] = {t: i for i, t in enumerate(self._vocabulary)}
        self._idf: Dict[str, float] = self._compute_idf(corpus)

    @property
    def vocabulary(self) -> Tuple[str, ...]:
        return self._vocabulary

    def _compute_idf(self, corpus: Sequence[Sequence[str]]) -> Dict[str, float]:
        total_documents = len(corpus)
        document_frequency: Dict[str, int] = {t: 0 for t in self._vocabulary}
        for document in corpus:
            for token in set(document):
                document_frequency[token] += 1
        return {
            t: math.log((total_documents + 1) / (document_frequency[t] + 1)) + 1.0
            for t in self._vocabulary
        }

    def _compute_tf(self, tokens: Sequence[str]) -> Dict[str, float]:
        total_terms = len(tokens)
        if total_terms == 0:
            return {}
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return {t: c / total_terms for t, c in counts.items()}

    def transform(self, tokens: Sequence[str]) -> List[float]:
        tf = self._compute_tf(tokens)
        vector = [0.0] * len(self._vocabulary)
        for token, tf_score in tf.items():
            position = self._index_of.get(token)
            if position is not None:
                vector[position] = tf_score * self._idf[token]
        return vector


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


class TechStackRecommender:
    """Content-Based Filtering engine implementing the 4-step pipeline:
    Ingestion -> Scoring -> Sorting -> Filtering."""

    def __init__(self, roles: List[JobRole]):
        if not roles:
            raise ValueError("Recommender requires at least one job role.")
        self._roles = roles
        self._vectorizer = TfidfVectorizer([role.skills for role in roles])
        self._role_vectors = [self._vectorizer.transform(role.skills) for role in roles]

    @property
    def known_skills(self) -> Tuple[str, ...]:
        return self._vectorizer.vocabulary

    @property
    def roles(self) -> Tuple[JobRole, ...]:
        return tuple(self._roles)

    def recommend(self, user_skills: Sequence[str], top_n: int = DEFAULT_TOP_N) -> List[Recommendation]:
        cleaned = tuple(sorted({normalize_token(s) for s in user_skills if s.strip()}))
        user_vector = self._vectorizer.transform(cleaned)

        scored: List[Recommendation] = []
        for role, role_vector in zip(self._roles, self._role_vectors):
            score = cosine_similarity(user_vector, role_vector)
            matched = tuple(s for s in cleaned if s in role.skills)
            scored.append(Recommendation(role=role, score=score, matched_skills=matched))

        scored.sort(key=lambda r: (-r.score, -len(r.matched_skills), r.role.name))
        return scored[: max(top_n, 1)]

    def trending_fallback(self, top_n: int = DEFAULT_TOP_N) -> List[Recommendation]:
        lookup = {role.name: role for role in self._roles}
        fallback_roles = [lookup[name] for name in TRENDING_FALLBACK_ROLES if name in lookup]
        if not fallback_roles:
            fallback_roles = list(self._roles[:top_n])
        return [Recommendation(role=r, score=0.0, matched_skills=()) for r in fallback_roles[:top_n]]
class RecommenderApp(tk.Tk):
    """Main application window."""

    BG_COLOR = "#0f172a"
    PANEL_COLOR = "#1e293b"
    ACCENT_COLOR = "#38bdf8"
    TEXT_COLOR = "#e2e8f0"
    MUTED_COLOR = "#94a3b8"
    SUCCESS_COLOR = "#4ade80"

    def __init__(self, engine: TechStackRecommender):
        super().__init__()
        self.engine = engine

        self.title("DecodeLabs | Tech Stack Recommender")
        self.geometry("980x640")
        self.minsize(860, 560)
        self.configure(bg=self.BG_COLOR)

        self._configure_styles()
        self._build_layout()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=self.BG_COLOR)
        style.configure("Panel.TFrame", background=self.PANEL_COLOR)
        style.configure("TLabel", background=self.BG_COLOR, foreground=self.TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=self.PANEL_COLOR, foreground=self.TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.BG_COLOR, foreground=self.TEXT_COLOR, font=("Segoe UI Semibold", 18))
        style.configure("Subtitle.TLabel", background=self.BG_COLOR, foreground=self.MUTED_COLOR, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=self.PANEL_COLOR, foreground=self.ACCENT_COLOR, font=("Segoe UI Semibold", 11))
        style.configure("Hint.TLabel", background=self.PANEL_COLOR, foreground=self.MUTED_COLOR, font=("Segoe UI", 9))

        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=8)
        style.map("Accent.TButton", background=[("active", "#0ea5e9")], foreground=[("active", "white")])

        style.configure(
            "Treeview",
            background="#0b1220",
            fieldbackground="#0b1220",
            foreground=self.TEXT_COLOR,
            rowheight=28,
            font=("Segoe UI", 10),
            borderwidth=0,
        )
        style.configure("Treeview.Heading", background=self.PANEL_COLOR, foreground=self.ACCENT_COLOR, font=("Segoe UI Semibold", 10))
        style.map("Treeview", background=[("selected", "#0ea5e9")])

        style.configure("TSpinbox", fieldbackground="#0b1220", foreground=self.TEXT_COLOR, arrowsize=14)
   
    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=24, pady=(20, 10))

        ttk.Label(header, text="Tech Stack Recommender", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="DecodeLabs AI Internship - Project 3  |  Content-Based Filtering (TF-IDF + Cosine Similarity)",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self, style="TFrame")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(0, weight=0, minsize=340)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_results_panel(body)

        self.status_var = tk.StringVar(value="Ready. Select skills and click 'Get Recommendations'.")
        status_bar = ttk.Label(self, textvariable=self.status_var, style="Subtitle.TLabel", anchor="w")
        status_bar.pack(fill="x", padx=24, pady=(0, 14))

    def _build_input_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="1. PICK SKILLS FROM THE LIST", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="Hold Ctrl / Cmd to select multiple", style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 6))

        list_frame = ttk.Frame(panel, style="Panel.TFrame")
        list_frame.grid(row=2, column=0, sticky="nsew")
        panel.rowconfigure(2, weight=1)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.skills_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            bg="#0b1220",
            fg=self.TEXT_COLOR,
            selectbackground=self.ACCENT_COLOR,
            selectforeground="#0b1220",
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
            height=12,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.skills_listbox.yview)
        for skill in self.engine.known_skills:
            self.skills_listbox.insert(tk.END, skill)
        self.skills_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(panel, text="2. OR TYPE SKILLS MANUALLY", style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=(14, 2))
        ttk.Label(panel, text="Comma-separated, e.g. python, automation, ui/ux", style="Hint.TLabel").grid(row=4, column=0, sticky="w")

        self.manual_entry = tk.Entry(
            panel, bg="#0b1220", fg=self.TEXT_COLOR, insertbackground=self.TEXT_COLOR,
            relief="flat", font=("Segoe UI", 10),
        )
        self.manual_entry.grid(row=5, column=0, sticky="ew", pady=(6, 14), ipady=6)

        options_row = ttk.Frame(panel, style="Panel.TFrame")
        options_row.grid(row=6, column=0, sticky="ew")
        options_row.columnconfigure(1, weight=1)

        ttk.Label(options_row, text="Recommendations:", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.top_n_var = tk.IntVar(value=DEFAULT_TOP_N)
        top_n_spin = ttk.Spinbox(
            options_row, from_=1, to=MAX_TOP_N, textvariable=self.top_n_var, width=4, justify="center"
        )
        top_n_spin.grid(row=0, column=1, sticky="w", padx=(8, 0))

        button_row = ttk.Frame(panel, style="Panel.TFrame")
        button_row.grid(row=7, column=0, sticky="ew", pady=(16, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        recommend_btn = ttk.Button(
            button_row, text="Get Recommendations", style="Accent.TButton", command=self.on_recommend_clicked
        )
        recommend_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        clear_btn = ttk.Button(button_row, text="Clear Selection", command=self.on_clear_clicked)
        clear_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.bind("<Return>", lambda _event: self.on_recommend_clicked())

    def _build_results_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        ttk.Label(panel, text="RANKED RECOMMENDATIONS", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        columns = ("rank", "role", "confidence", "matched", "full_skillset")
        self.results_tree = ttk.Treeview(panel, columns=columns, show="headings", selectmode="browse")

        headings = {
            "rank": ("#", 40),
            "role": ("Job Role", 160),
            "confidence": ("Match %", 80),
            "matched": ("Matched Skills", 220),
            "full_skillset": ("Full Skill Set", 300),
        }
        for column_id, (label, width) in headings.items():
            self.results_tree.heading(column_id, text=label)
            self.results_tree.column(column_id, width=width, anchor="w", stretch=(column_id == "full_skillset"))

        v_scroll = ttk.Scrollbar(panel, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=v_scroll.set)

        self.results_tree.grid(row=1, column=0, sticky="nsew")
        v_scroll.grid(row=1, column=1, sticky="ns")

        self.results_tree.tag_configure("top1", background="#0f2a1e", foreground=self.SUCCESS_COLOR)
        self.results_tree.tag_configure("fallback", background="#2a1f0f", foreground="#facc15")

    def _collect_selected_skills(self) -> List[str]:
        listbox_skills = [self.skills_listbox.get(i) for i in self.skills_listbox.curselection()]
        manual_skills = parse_skill_input(self.manual_entry.get())

        combined: Dict[str, None] = {}
        for skill in listbox_skills + manual_skills:
            combined.setdefault(normalize_token(skill), None)
        return list(combined.keys())

    def on_recommend_clicked(self) -> None:
        skills = self._collect_selected_skills()

        if len(skills) < MIN_REQUIRED_SKILLS:
            messagebox.showwarning(
                "More Skills Needed",
                f"Please select or type at least {MIN_REQUIRED_SKILLS} distinct skills.\n"
                f"You currently have {len(skills)}.",
            )
            self.status_var.set(f"[!] Need at least {MIN_REQUIRED_SKILLS} skills - you provided {len(skills)}.")
            return

        top_n = self._safe_top_n()
        recommendations = self.engine.recommend(skills, top_n=top_n)
        best_score = recommendations[0].score if recommendations else 0.0

        is_fallback = best_score <= 0.0
        if is_fallback:
            recommendations = self.engine.trending_fallback(top_n=top_n)

        self._render_results(recommendations, is_fallback=is_fallback)

        if is_fallback:
            self.status_var.set(
                "No vocabulary overlap found for those skills - showing trending roles instead (cold start)."
            )
        else:
            self.status_var.set(f"Showing top {len(recommendations)} recommendation(s) for {len(skills)} skill(s).")

    def on_clear_clicked(self) -> None:
        self.skills_listbox.selection_clear(0, tk.END)
        self.manual_entry.delete(0, tk.END)
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)
        self.status_var.set("Selection cleared. Ready for new input.")

    def _safe_top_n(self) -> int:
        try:
            value = int(self.top_n_var.get())
        except (tk.TclError, ValueError):
            value = DEFAULT_TOP_N
        return max(1, min(value, MAX_TOP_N))

    def _render_results(self, recommendations: List[Recommendation], is_fallback: bool) -> None:
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)

        for rank, rec in enumerate(recommendations, start=1):
            confidence_text = "trending" if is_fallback else f"{rec.confidence_percent}%"
            matched_text = ", ".join(rec.matched_skills) if rec.matched_skills else "-"
            full_skillset_text = ", ".join(rec.role.skills)

            tag = "fallback" if is_fallback else ("top1" if rank == 1 else "")
            self.results_tree.insert(
                "", tk.END,
                values=(rank, rec.role.name, confidence_text, matched_text, full_skillset_text),
                tags=(tag,) if tag else (),
            )

def main() -> None:
    roles = load_job_roles()
    engine = TechStackRecommender(roles)

    app = RecommenderApp(engine)
    app.mainloop()


if __name__ == "__main__":
    main()
