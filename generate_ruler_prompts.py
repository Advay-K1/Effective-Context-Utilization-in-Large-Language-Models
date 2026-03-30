#!/usr/bin/env python3
"""
generate_ruler_prompts.py
=========================
Generate RULER-style needle-in-a-haystack prompts for evaluating
effective context utilization in long-context LLMs.

Each prompt hides a unique retrievable fact (the "needle") at a specified
fractional position within a long filler passage (the "haystack").
A question about the needle is appended at the end.

Output JSONL format (matches chunk_deletion_baseline.py):
    {"prompt": "...", "reference": "...", "task": "niah"}

Usage:
    # All combinations of lengths and positions, 5 examples each
    python generate_ruler_prompts.py \\
        --lengths 2k 4k 8k 16k \\
        --positions beginning middle end \\
        --num_examples 5 \\
        --output_file ruler_prompts.jsonl

    # Single config
    python generate_ruler_prompts.py \\
        --lengths 8k \\
        --positions middle \\
        --num_examples 10 \\
        --output_file prompts_8k_mid.jsonl

    # Use an actual tokenizer for precise length targeting
    python generate_ruler_prompts.py \\
        --lengths 4k 8k \\
        --positions beginning middle end \\
        --num_examples 3 \\
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
        --output_file ruler_prompts.jsonl
"""

import argparse
import json
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4  # default approximation for English text


def make_token_counter(tokenizer_name: Optional[str]):
    """Return a function that estimates token count for a string.

    If tokenizer_name is given and the model is loadable, uses the real
    HuggingFace tokenizer. Otherwise falls back to chars / 4.
    """
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"Using tokenizer: {tokenizer_name}", file=sys.stderr)

            def count_tokens(text: str) -> int:
                return len(tok.encode(text, add_special_tokens=False))

            return count_tokens
        except Exception as e:
            print(
                f"Warning: could not load tokenizer '{tokenizer_name}' ({e}). "
                "Falling back to chars/4 approximation.",
                file=sys.stderr,
            )

    def count_tokens_approx(text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    return count_tokens_approx


def parse_length(s: str) -> int:
    """Parse '2k' -> 2048, '4k' -> 4096, etc. Also accepts plain integers."""
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k", s)
    if m:
        return int(float(m.group(1)) * 1024)
    return int(s)


# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------

POSITION_FRACTIONS = {
    "beginning": 0.1,
    "middle":    0.5,
    "end":       0.9,
}


# ---------------------------------------------------------------------------
# Haystack paragraph pool
# ---------------------------------------------------------------------------
# ~30 varied paragraphs, each 60-120 words (~240-480 chars, ~60-120 tokens).
# Covering different domains for diversity.

HAYSTACK_PARAGRAPHS = [
    # Science
    "Photosynthesis is the process by which plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in glucose. In the chloroplasts, chlorophyll "
    "absorbs sunlight and drives the conversion of carbon dioxide and water into sugar and "
    "oxygen. This process forms the foundation of nearly all food chains on Earth and is "
    "responsible for producing the oxygen in our atmosphere.",

    "The theory of plate tectonics describes how Earth's outer shell is divided into "
    "several large plates that glide over the mantle. These plates move a few centimeters "
    "per year, driven by convection currents in the hot mantle rock below. When plates "
    "collide, separate, or slide past each other, the result can be earthquakes, volcanic "
    "eruptions, and the formation of mountain ranges over geological timescales.",

    "Black holes form when massive stars exhaust their nuclear fuel and collapse under "
    "their own gravity. The resulting object has a gravitational pull so strong that nothing, "
    "not even light, can escape once past the event horizon. Astronomers detect them "
    "indirectly through their effects on nearby matter and through gravitational waves "
    "produced when two black holes merge.",

    "The human immune system is a complex network of cells, tissues, and organs that "
    "defends the body against pathogens. It distinguishes between self and non-self using "
    "surface proteins called antigens. When foreign antigens are detected, B cells produce "
    "antibodies and T cells launch a targeted attack. Memory cells allow the body to "
    "respond more quickly to pathogens encountered previously.",

    "Quantum entanglement occurs when two particles become correlated in such a way that "
    "the quantum state of each cannot be described independently of the other. Measuring "
    "a property of one particle instantaneously influences the corresponding property of "
    "its entangled partner, regardless of the distance between them. This phenomenon has "
    "potential applications in quantum computing and secure communications.",

    # History
    "The Silk Road was a network of trade routes that connected China with Central Asia, "
    "the Middle East, and Europe for roughly fifteen centuries. Merchants carried silk, "
    "spices, glassware, and ideas across deserts and mountain passes. The exchange of "
    "goods along these routes also facilitated the spread of religions, languages, and "
    "technologies between civilizations that would otherwise have had little contact.",

    "The printing press, invented by Johannes Gutenberg around 1440, transformed the "
    "dissemination of knowledge in Europe. Before its invention, books were copied by "
    "hand, making them rare and expensive. The press made it possible to produce texts "
    "quickly and cheaply, accelerating literacy, the Protestant Reformation, and the "
    "Scientific Revolution by making ideas available to far wider audiences.",

    "The Industrial Revolution began in Britain in the late eighteenth century and spread "
    "across Europe and North America over the following decades. Steam-powered machinery "
    "replaced manual labor in textile mills and ironworks, driving urbanization and "
    "fundamentally changing social structures. New forms of transport, including railways "
    "and steamships, shrank distances and created the first truly global economy.",

    "Ancient Rome grew from a small city-state on the Tiber River into an empire that "
    "encompassed much of Europe, North Africa, and the Near East. Roman law, engineering, "
    "language, and governance influenced the development of Western civilization for "
    "millennia. The empire's eventual fragmentation in the fifth century gave rise to "
    "medieval European kingdoms and the Byzantine Empire in the east.",

    "The Renaissance was a cultural and intellectual movement that began in Italy during "
    "the fourteenth century and spread throughout Europe over the next two hundred years. "
    "Artists and scholars revived classical Greek and Roman texts and ideas, producing "
    "remarkable advances in painting, sculpture, architecture, and literature. Figures "
    "such as Leonardo da Vinci and Michelangelo exemplified the era's celebration of human "
    "achievement and inquiry.",

    # Geography
    "The Amazon Basin covers approximately seven million square kilometers across nine "
    "South American countries, making it the world's largest tropical rainforest. It "
    "houses roughly ten percent of all species on Earth and plays a critical role in "
    "regulating the global climate by absorbing vast amounts of carbon dioxide. The "
    "Amazon River discharges about twenty percent of all fresh water entering the world's "
    "oceans.",

    "The Sahara is the world's largest hot desert, stretching more than nine million "
    "square kilometers across northern Africa. Despite its reputation for endless sand "
    "dunes, much of the Sahara consists of rocky plateaus, gravel plains, and mountain "
    "ranges. Temperatures can exceed fifty degrees Celsius by day and plummet below "
    "freezing at night. Scattered oases support communities that have traded across the "
    "desert for thousands of years.",

    "The Himalayas form the world's highest mountain range, running more than "
    "twenty-four hundred kilometers along the borders of Nepal, India, Bhutan, Tibet, "
    "and Pakistan. The range was created by the collision of the Indian and Eurasian "
    "tectonic plates and continues to rise by a few millimeters each year. It contains "
    "fourteen peaks above eight thousand meters, including Mount Everest.",

    "The Pacific Ocean covers more area than all of Earth's landmasses combined and "
    "contains more than half of the world's oceanic water. It stretches from the Arctic "
    "in the north to Antarctica in the south and is bounded by Asia and Australia to "
    "the west and the Americas to the east. The Ring of Fire, a zone of intense seismic "
    "and volcanic activity, follows its edges.",

    "The Great Barrier Reef, located off the northeastern coast of Australia, is the "
    "world's largest coral reef system. It stretches over two thousand kilometers and "
    "is home to an extraordinary diversity of marine life, including fish, sea turtles, "
    "dolphins, and thousands of coral species. Rising ocean temperatures and acidification "
    "caused by climate change pose a serious threat to its long-term survival.",

    # Technology
    "The internet originated in the late 1960s as ARPANET, a United States military "
    "research network designed to survive a nuclear attack. Over subsequent decades, "
    "universities and research institutions joined the network. The invention of the "
    "World Wide Web by Tim Berners-Lee in 1989 transformed it into a global platform "
    "for communication, commerce, and information sharing accessible to ordinary people.",

    "Artificial neural networks are computational models loosely inspired by the structure "
    "of the human brain. They consist of layers of interconnected nodes that process "
    "input data and adjust connection weights during training. Deep learning, which uses "
    "networks with many hidden layers, has achieved breakthroughs in image recognition, "
    "natural language processing, and game-playing over the past decade.",

    "The transistor, invented at Bell Labs in 1947, replaced vacuum tubes and became "
    "the fundamental building block of modern electronics. Integrated circuits pack "
    "billions of transistors onto a sliver of silicon smaller than a fingernail. This "
    "miniaturization has made possible smartphones, laptops, and countless other devices "
    "that would have seemed like science fiction to the engineers who built the first "
    "computers.",

    "GPS, or the Global Positioning System, uses a constellation of at least twenty-four "
    "satellites orbiting Earth to provide location and time information anywhere on the "
    "planet. A receiver calculates its position by measuring the time it takes signals "
    "to arrive from multiple satellites. Originally developed for military use, GPS now "
    "underpins navigation, agriculture, emergency response, and financial transaction "
    "timing.",

    "Cryptography is the practice of securing information so that only intended recipients "
    "can read it. Modern cryptographic systems rely on mathematical problems that are "
    "easy to compute in one direction but practically impossible to reverse. Public-key "
    "cryptography, developed in the 1970s, allows two parties to communicate securely "
    "without having previously shared a secret key, forming the basis of secure internet "
    "transactions.",

    # Biology
    "DNA, or deoxyribonucleic acid, carries the genetic instructions for the development "
    "and function of all known living organisms. It consists of two strands wound into "
    "a double helix, with bases on each strand pairing complementarily. The sequence of "
    "four bases — adenine, thymine, cytosine, and guanine — encodes the instructions for "
    "building proteins, which carry out nearly every cellular function.",

    "Evolution by natural selection, described by Charles Darwin in 1859, explains the "
    "diversity of life on Earth. Organisms with traits better suited to their environment "
    "survive and reproduce more successfully, passing those traits to offspring. Over many "
    "generations, populations change, and new species emerge. The fossil record and modern "
    "genetics provide converging lines of evidence supporting this theory.",

    "Neurons are the basic signaling cells of the nervous system. They transmit information "
    "as electrical impulses along their axons and release chemical neurotransmitters at "
    "synapses to communicate with neighboring cells. The human brain contains roughly "
    "eighty-six billion neurons, each forming thousands of connections. This network "
    "underlies perception, thought, memory, and all voluntary movement.",

    "Viruses are not considered living organisms because they cannot replicate on their "
    "own. Instead, they inject their genetic material into host cells and hijack the "
    "cell's machinery to produce copies of themselves. The new virus particles then burst "
    "out and infect neighboring cells. This process can cause diseases ranging from the "
    "common cold to influenza, HIV, and COVID-19.",

    # Economics / Society
    "Supply and demand is the foundational model of market economics. When supply of a "
    "good exceeds demand, prices fall; when demand exceeds supply, prices rise. Prices "
    "thus act as signals that coordinate the decisions of millions of buyers and sellers "
    "without central direction. The model underlies most economic analysis of markets, "
    "trade, and policy.",

    "Urbanization, the movement of populations from rural to urban areas, has accelerated "
    "dramatically since the Industrial Revolution. Today, more than half of the world's "
    "population lives in cities, a proportion expected to rise to two-thirds by 2050. "
    "Urban areas drive economic growth and innovation but also concentrate pollution, "
    "inequality, and infrastructure challenges.",

    "Climate change refers to long-term shifts in global temperatures and weather patterns "
    "caused primarily by human emissions of greenhouse gases since the Industrial "
    "Revolution. Rising temperatures are melting ice sheets, raising sea levels, and "
    "intensifying extreme weather events. International agreements such as the Paris "
    "Accord aim to limit warming, but emissions continue to rise globally.",

    "The gig economy describes labor markets characterized by short-term contracts and "
    "freelance work rather than permanent employment. Platforms such as ride-sharing "
    "and delivery services have expanded this model significantly. While gig work offers "
    "flexibility, workers typically lack benefits such as health insurance and retirement "
    "savings, raising questions about labor protections in the digital age.",

    # Mathematics / Logic
    "Prime numbers are integers greater than one that have no divisors other than one "
    "and themselves. They are the building blocks of all integers via the fundamental "
    "theorem of arithmetic, which states that every integer greater than one is either "
    "prime or can be expressed as a unique product of primes. Despite their simple "
    "definition, primes exhibit patterns that mathematicians have studied for millennia.",

    "Statistics is the science of collecting, analyzing, and interpreting data. Descriptive "
    "statistics summarize datasets using measures such as mean, median, and standard "
    "deviation. Inferential statistics allow conclusions about a larger population to be "
    "drawn from a sample, using probability theory to quantify uncertainty. It is used "
    "across science, medicine, social research, and industry.",

    "Graph theory studies structures made of vertices connected by edges. It underpins "
    "network analysis across disciplines: social networks, road maps, the internet, and "
    "molecular structures can all be modeled as graphs. Famous problems in graph theory "
    "include finding the shortest path between two nodes and determining whether a graph "
    "can be colored with a minimum number of colors so no adjacent nodes share a color.",
]


# ---------------------------------------------------------------------------
# Needle templates
# ---------------------------------------------------------------------------
# Each entry: (statement_template, question_template)
# Placeholders: {label} — a unique project/item identifier; {value} — the code to retrieve.

NEEDLE_TEMPLATES = [
    (
        "The secret access code for project {label} is \"{value}\".",
        "What is the secret access code for project {label}?",
    ),
    (
        "The authentication token assigned to experiment {label} is \"{value}\".",
        "What is the authentication token assigned to experiment {label}?",
    ),
    (
        "The unique identifier registered for operation {label} is \"{value}\".",
        "What is the unique identifier registered for operation {label}?",
    ),
    (
        "The passphrase used to unlock vault {label} is \"{value}\".",
        "What is the passphrase used to unlock vault {label}?",
    ),
    (
        "The verification key for mission {label} is \"{value}\".",
        "What is the verification key for mission {label}?",
    ),
    (
        "The conference room booking code for event {label} is \"{value}\".",
        "What is the conference room booking code for event {label}?",
    ),
    (
        "The recovery phrase for account {label} is \"{value}\".",
        "What is the recovery phrase for account {label}?",
    ),
    (
        "The serial number assigned to unit {label} is \"{value}\".",
        "What is the serial number assigned to unit {label}?",
    ),
]

# Project/item label pool for the needle
LABEL_PREFIXES = [
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA",
    "IOTA", "KAPPA", "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI", "RHO",
    "SIGMA", "TAU", "UPSILON", "PHI", "CHI", "PSI", "OMEGA",
    "AURORA", "BEACON", "COBALT", "DRIFTER", "ECLIPSE", "FALCON",
    "GRANITE", "HARBOR", "INDIGO", "JASPER", "KESTREL", "LANCER",
    "MARBLE", "NEBULA", "OCELOT", "PHANTOM", "QUARTZ", "RAPTOR",
    "STELLAR", "TITAN", "UMBRA", "VORTEX", "WARDEN", "XENON",
    "YELLOW", "ZEPHYR",
]


def generate_label(rng: random.Random, used: set) -> str:
    """Pick a unique label from the pool, appending a number if needed."""
    candidates = list(LABEL_PREFIXES)
    rng.shuffle(candidates)
    for prefix in candidates:
        if prefix not in used:
            used.add(prefix)
            return prefix
    # Fall back to numbered labels
    n = len(used)
    label = f"PROJECT-{n:04d}"
    used.add(label)
    return label


def generate_code(rng: random.Random) -> str:
    """Generate a distinctive random code unlikely to appear in the haystack.

    Format: XXXX-YYYY-ZZZZ where each segment is alphanumeric.
    Example: K7M2-PLFT-9XQW
    """
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I/O/1/0 to avoid confusion
    seg = lambda n: "".join(rng.choices(chars, k=n))
    return f"{seg(4)}-{seg(4)}-{seg(4)}"


# ---------------------------------------------------------------------------
# Haystack builder
# ---------------------------------------------------------------------------

def build_haystack_paragraphs(
    target_chars: int,
    rng: random.Random,
) -> list[str]:
    """Return a shuffled list of paragraphs whose total char count is approximately
    target_chars. Paragraphs are drawn with replacement from the pool if needed."""
    pool = list(HAYSTACK_PARAGRAPHS)
    rng.shuffle(pool)

    result: list[str] = []
    total = 0
    pool_index = 0

    while total < target_chars:
        para = pool[pool_index % len(pool)]
        pool_index += 1
        # Avoid consecutive duplicates
        if result and result[-1] == para:
            continue
        result.append(para)
        total += len(para) + 2  # +2 for "\n\n" separator

    return result


def insert_needle_paragraph(
    paragraphs: list[str],
    needle_text: str,
    position_fraction: float,
) -> list[str]:
    """Insert the needle as a paragraph at the given fractional position."""
    insert_at = max(0, min(len(paragraphs), round(position_fraction * len(paragraphs))))
    return paragraphs[:insert_at] + [needle_text] + paragraphs[insert_at:]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

PROMPT_PREAMBLE = (
    "Read the following text carefully. "
    "A specific piece of information is hidden somewhere in the text. "
    "Answer the question at the end based solely on what you find in the text.\n\n"
)


def assemble_prompt(paragraphs: list[str], question: str) -> str:
    body = "\n\n".join(paragraphs)
    return f"{PROMPT_PREAMBLE}{body}\n\nQuestion: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def generate_example(
    target_tokens: int,
    position: str,
    rng: random.Random,
    count_tokens,
    used_labels: set,
) -> dict:
    """Generate one NIAH example."""
    position_fraction = POSITION_FRACTIONS[position]

    # Pick a needle template and generate unique label + code
    template_stmt, template_q = rng.choice(NEEDLE_TEMPLATES)
    label = generate_label(rng, used_labels)
    code = generate_code(rng)

    needle_text = template_stmt.format(label=label, value=code)
    question = template_q.format(label=label)

    # Estimate chars needed for haystack
    preamble_chars = len(PROMPT_PREAMBLE)
    needle_chars = len(needle_text) + 2
    question_chars = len(f"\n\nQuestion: {question}\nAnswer:") + 2
    target_chars = target_tokens * CHARS_PER_TOKEN
    haystack_chars = max(100, target_chars - preamble_chars - needle_chars - question_chars)

    # Build and insert needle
    paras = build_haystack_paragraphs(haystack_chars, rng)
    paras = insert_needle_paragraph(paras, needle_text, position_fraction)

    prompt = assemble_prompt(paras, question)

    # Trim or pad to get closer to target length
    # (simple trim: if over by >20%, remove trailing haystack paragraphs)
    actual = count_tokens(prompt)
    if actual > target_tokens * 1.15:
        # Remove paragraphs from the end (not the needle) until within budget
        while len(paras) > 2:
            needle_idx = paras.index(needle_text)
            # Find a removable paragraph at the end that isn't the needle
            if paras[-1] != needle_text:
                paras.pop()
            elif paras[0] != needle_text:
                paras.pop(0)
            else:
                break
            prompt = assemble_prompt(paras, question)
            if count_tokens(prompt) <= target_tokens * 1.1:
                break

    return {
        "prompt": prompt,
        "reference": code,
        "task": "niah",
        "metadata": {
            "target_tokens": target_tokens,
            "actual_tokens": count_tokens(prompt),
            "position": position,
            "needle_label": label,
            "needle_fraction": position_fraction,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate RULER-style needle-in-a-haystack prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lengths", nargs="+", default=["2k", "4k", "8k", "16k"],
        help="Target prompt lengths (e.g. 2k 4k 8k 16k or plain token counts). "
             "Default: 2k 4k 8k 16k",
    )
    parser.add_argument(
        "--positions", nargs="+", default=["beginning", "middle", "end"],
        choices=["beginning", "middle", "end"],
        help="Needle positions within the haystack. Default: beginning middle end",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5,
        help="Number of examples per (length, position) combination. Default: 5",
    )
    parser.add_argument(
        "--output_file", type=str, default="ruler_prompts.jsonl",
        help="Output JSONL file path. Default: ruler_prompts.jsonl",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="HuggingFace tokenizer name/path for precise token counting. "
             "If omitted, uses chars/4 approximation.",
    )
    parser.add_argument(
        "--include_metadata", action="store_true",
        help="Include a 'metadata' field in each output record.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    count_tokens = make_token_counter(args.tokenizer)

    lengths = [parse_length(l) for l in args.lengths]
    combos = list(product(lengths, args.positions))
    total = len(combos) * args.num_examples

    print(
        f"Generating {total} examples "
        f"({len(lengths)} lengths × {len(args.positions)} positions × {args.num_examples} examples)",
        file=sys.stderr,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    used_labels: set = set()

    with out_path.open("w") as f:
        for target_tokens, position in combos:
            for i in range(args.num_examples):
                ex = generate_example(
                    target_tokens=target_tokens,
                    position=position,
                    rng=rng,
                    count_tokens=count_tokens,
                    used_labels=used_labels,
                )
                record = {
                    "prompt": ex["prompt"],
                    "reference": ex["reference"],
                    "task": ex["task"],
                }
                if args.include_metadata:
                    record["metadata"] = ex["metadata"]

                f.write(json.dumps(record) + "\n")
                written += 1

                actual = ex["metadata"]["actual_tokens"]
                print(
                    f"  [{written}/{total}] length={target_tokens} position={position} "
                    f"actual_tokens≈{actual} label={ex['metadata']['needle_label']}",
                    file=sys.stderr,
                )

    print(f"\nWrote {written} examples to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
