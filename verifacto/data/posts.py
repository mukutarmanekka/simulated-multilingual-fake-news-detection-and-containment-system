"""Step 2: the post corpus.

Each post has `text`, a hidden `true_label` ("fake" or "real"), an `author`
node id, and a `language`. The label is ground truth for METRICS ONLY — the
detector and the intervention logic never see it.

Posts are generated from hand-written claim templates per language:
12 real + 12 fake templates in en, hi, fr. Placeholders ({city}, {percent},
{country}) are filled from per-language pools with a seeded rng, so every
generated post is a distinct, checkable claim.
"""

import json
import string
from dataclasses import asdict, dataclass

import numpy as np

from ..config import LANGUAGES, N_POSTS, POSTS_PATH, POSTS_SEED, REAL_FRACTION
from ..network.graph_builder import block_node_lists

# ── Claim templates ──────────────────────────────────────────────────
# `true_label` for each template is known by construction: the REAL set
# contains only verifiable true claims, the FAKE set only plausible
# misinformation. The simulator uses these labels for metrics only.

REAL_TEMPLATES = {
    "en": [
        "The Eiffel Tower is located in Paris, France.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Moon orbits the Earth once about every 27 days.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Light from the Sun takes about 8 minutes to reach Earth.",
        "India gained independence from British rule in 1947.",
        "The human body has 206 bones.",
        "Mount Everest is the highest mountain above sea level.",
        "Honey bees communicate the location of flowers through a waggle dance.",
        "The Great Wall of China is over 20,000 kilometers long.",
        "Antarctica is the coldest continent on Earth.",
        "Vaccines train the immune system to recognize specific pathogens.",
        "The Amazon River is the largest river in the world by water volume.",
        "Tokyo is the capital city of Japan.",
    ],
    "hi": [
        "भारत को 1947 में ब्रिटिश शासन से स्वतंत्रता मिली।",
        "सूर्य से प्रकाश को पृथ्वी तक पहुंचने में लगभग 8 मिनट लगते हैं।",
        "मानव शरीर में 206 हड्डियां होती हैं।",
        "चंद्रमा पृथ्वी की परिक्रमा करता है।",
        "माउंट एवरेस्ट समुद्र तल से सबसे ऊंचा पर्वत है।",
        "ताजमहल आगरा में स्थित है।",
        "समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।",
        "हिमालय भारत के उत्तर में स्थित एक पर्वत श्रृंखला है।",
        "चंद्रयान-3 मिशन ने चंद्रमा के दक्षिणी ध्रुव के पास सफलतापूर्वक लैंडिंग की।",
        "गंगा भारत की सबसे लंबी नदी है।",
        "टीके प्रतिरक्षा प्रणाली को बीमारियों से लड़ने के लिए प्रशिक्षित करते हैं।",
        "पृथ्वी सूर्य की परिक्रमा करती है।",
        "अमेज़न नदी जल की मात्रा के हिसाब से दुनिया की सबसे बड़ी नदी है।",
        "टोक्यो जापान की राजधानी है।",
    ],
    "fr": [
        "La tour Eiffel est située à Paris, en France.",
        "L'eau bout à 100 degrés Celsius au niveau de la mer.",
        "La Lune orbite autour de la Terre.",
        "L'océan Pacifique est le plus grand océan de la planète.",
        "La lumière du Soleil met environ 8 minutes pour atteindre la Terre.",
        "L'Everest est la plus haute montagne au-dessus du niveau de la mer.",
        "Le corps humain compte 206 os.",
        "Les abeilles communiquent par une danse frétillante.",
        "Le Taj Mahal se trouve à Agra, en Inde.",
        "L'Antarctique est le continent le plus froid de la Terre.",
        "La Grande Muraille de Chine mesure plus de 20 000 kilomètres.",
        "Les vaccins entraînent le système immunitaire à reconnaître des agents pathogènes.",
        "L'Amazone est le plus grand fleuve du monde par son débit.",
        "Tokyo est la capitale du Japon.",
    ],
}

FAKE_TEMPLATES = {
    "en": [
        "NASA has confirmed that liquid water flows freely on the surface of the Moon.",
        "A new study shows that {percent}% of smartphone batteries explode within their first year of use.",
        "Scientists at {city} University have proven that 5G towers cause widespread headaches and insomnia.",
        "The government of {country} has banned all private internet access starting next month.",
        "Doctors now recommend eating raw garlic as a complete cure for diabetes.",
        "A leaked report reveals that the Earth is actually flat and space agencies have hidden the evidence.",
        "Electric cars cause more pollution than diesel trucks because of battery production.",
        "The WHO has officially declared that video games are a form of digital drug addiction.",
        "Billions of birds have gone blind worldwide because of solar panels reflecting sunlight.",
        "A hospital in {city} has recorded hundreds of deaths caused by the common flu vaccine.",
        "The pyramids of Egypt were built in the 20th century by modern construction companies.",
        "Drinking warm water every morning dissolves blood clots and prevents all heart attacks.",
        "A {city} laboratory has created a free energy machine that runs forever without any fuel.",
        "The World Health Organization has announced that {percent}% of the global population is immune to all viruses without any vaccine.",
    ],
    "hi": [
        "नासा ने पुष्टि की है कि चंद्रमा की सतह पर तरल पानी की नदियां बहती हैं।",
        "एक नए अध्ययन के अनुसार {percent}% स्मार्टफोन एक साल के भीतर फट जाते हैं।",
        "{city} विश्वविद्यालय के वैज्ञानिकों ने साबित किया कि 5G टावरों से लोग बीमार पड़ रहे हैं।",
        "सरकार ने अगले महीने से सभी निजी इंटरनेट सेवाओं पर प्रतिबंध लगा दिया है।",
        "डॉक्टरों का कहना है कि कच्चा लहसुन खाने से मधुमेह पूरी तरह ठीक हो जाता है।",
        "एक लीक रिपोर्ट से पता चला है कि पृथ्वी असल में सपाट है।",
        "इलेक्ट्रिक कारें डीजल ट्रकों से ज्यादा प्रदूषण फैलाती हैं।",
        "विश्व स्वास्थ्य संगठन ने वीडियो गेम को डिजिटल नशा घोषित कर दिया है।",
        "सोलर पैनल की चमक से दुनिया भर के पक्षी अंधे हो रहे हैं।",
        "{city} के अस्पताल में फ्लू के टीके से सैकड़ों मौतें दर्ज हुई हैं।",
        "मिस्र के पिरामिड 20वीं सदी में आधुनिक कंपनियों ने बनाए थे।",
        "रोज सुबह गर्म पानी पीने से सभी दिल के दौरे रुक जाते हैं।",
        "{city} की एक प्रयोगशाला ने ऐसी मुफ्त ऊर्जा मशीन बनाई है जो बिना ईंधन के हमेशा चलती है।",
        "विश्व स्वास्थ्य संगठन ने घोषणा की है कि दुनिया की {percent}% आबादी बिना टीके के ही सभी वायरस से प्रतिरक्षित है।",
    ],
    "fr": [
        "La NASA a confirmé que de l'eau liquide coule librement à la surface de la Lune.",
        "Une nouvelle étude montre que {percent} % des batteries de smartphones explosent dès la première année.",
        "Des scientifiques de l'université de {city} ont prouvé que les antennes 5G provoquent des insomnies.",
        "Le gouvernement de {country} a interdit tout accès internet privé dès le mois prochain.",
        "Les médecins recommandent l'ail cru comme remède complet contre le diabète.",
        "Un rapport divulgué révèle que la Terre est en réalité plate.",
        "Les voitures électriques polluent plus que les camions diesel.",
        "L'OMS a officiellement classé les jeux vidéo comme une drogue numérique.",
        "Des milliards d'oiseaux sont devenus aveugles à cause des panneaux solaires.",
        "Un hôpital de {city} a enregistré des centaines de décès dus au vaccin contre la grippe.",
        "Les pyramides d'Égypte ont été construites au 20e siècle par des entreprises modernes.",
        "Boire de l'eau chaude chaque matin dissout les caillots sanguins et prévient les crises cardiaques.",
        "Un laboratoire de {city} a créé une machine à énergie gratuite qui fonctionne sans carburant.",
        "L'OMS a annoncé que {percent} % de la population mondiale est immunisée contre tous les virus sans aucun vaccin.",
    ],
}

# ── Placeholder pools ────────────────────────────────────────────────

PLACEHOLDER_POOLS = {
    "en": {
        "city": ["Austin", "Leeds", "Osaka", "Toronto", "Cairo", "Seoul", "Lima", "Zurich"],
        "country": ["Brazil", "Japan", "Kenya", "Norway"],
        "percent": [35, 48, 62, 74, 87],
    },
    "hi": {
        "city": ["पटना", "भोपाल", "जयपुर", "लखनऊ", "नागपुर", "सूरत"],
        "country": ["भारत", "नेपाल", "श्रीलंका"],
        "percent": [35, 48, 62, 74, 87],
    },
    "fr": {
        "city": ["Lyon", "Toulouse", "Bruxelles", "Genève", "Nice", "Nantes"],
        "country": ["la France", "la Belgique", "le Canada", "la Suisse"],
        "percent": [35, 48, 62, 74, 87],
    },
}


@dataclass
class Post:
    id: int
    text: str
    true_label: str  # "fake" or "real" — hidden ground truth, metrics only
    author: int      # node id in G
    language: str    # ISO 639-1 code, matches the author's block
    source_url: str | None = None  # optional publisher URL for Tier 4 checking


def _fill_placeholders(template, language, rng):
    """Fill {placeholder} fields of a template from the language pools."""
    values = {}
    for _, field, _, _ in string.Formatter().parse(template):
        if field is None:
            continue
        values[field] = str(rng.choice(PLACEHOLDER_POOLS[language][field]))
    return template.format(**values)


def generate_posts(G, n_total=N_POSTS, real_fraction=REAL_FRACTION, seed=POSTS_SEED):
    """Generate the post corpus deterministically.

    Posts alternate real/fake and cycle through the languages, so the
    corpus is balanced: with the defaults this gives 40 real + 40 fake
    spread over en/hi/fr. Each post's author is a node in its language
    block.
    """
    templates = {"fake": FAKE_TEMPLATES, "real": REAL_TEMPLATES}
    rng = np.random.default_rng(seed)
    block_lists = block_node_lists(G)
    # each (label, language) group recurs every `period` ids; indexing by
    # post_id // period gives every slot its own template (14 per group)
    period = 2 * len(LANGUAGES)
    posts = []
    for post_id in range(n_total):
        language = LANGUAGES[post_id % len(LANGUAGES)]
        true_label = "real" if post_id % 2 == 0 else "fake"
        group = templates[true_label][language]
        template = group[(post_id // period) % len(group)]
        text = _fill_placeholders(template, language, rng)
        author = int(rng.choice(block_lists[LANGUAGES.index(language)]))
        posts.append(
            Post(id=post_id, text=text, true_label=true_label, author=author, language=language)
        )
    return posts


def save_posts(posts, path=POSTS_PATH) -> None:
    payload = {
        "meta": {"n_total": len(posts), "seed": POSTS_SEED, "languages": LANGUAGES},
        "posts": [asdict(p) for p in posts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_posts(path=POSTS_PATH) -> list[Post]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [Post(**item) for item in payload["posts"]]


def print_posts_summary(posts) -> None:
    print("=== Post corpus summary ===")
    print(f"total posts: {len(posts)}")
    by_language = {language: {"real": 0, "fake": 0} for language in LANGUAGES}
    for post in posts:
        by_language[post.language][post.true_label] += 1
    for language in LANGUAGES:
        counts = by_language[language]
        total = counts["real"] + counts["fake"]
        print(f"  {language}: {total} posts ({counts['real']} real, {counts['fake']} fake)")
    n_real = sum(1 for p in posts if p.true_label == "real")
    n_fake = len(posts) - n_real
    print(f"real/fake ratio: {n_real}/{n_fake}")
