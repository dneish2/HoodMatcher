Key Technical Challenges and Solutions
⸻

1. Home Pricing Retrieval — Transition from Fuzzy Matching to RAG
s
Problem:
	•	Initially used fuzzy string matching (difflib) combined with normalization (lowercasing) to match neighborhood names.
	•	This method struggled with inconsistencies and didn’t scale well to messier, real-world datasets.

Reasoning and Decision:
	•	Shifted toward a Retrieval-Augmented Generation (RAG) architecture.
	•	RAG provided true semantic matching, ensuring broader robustness across diverse data variations with minimal additional engineering overhead.

⸻

2. Zillow Dataset Integration — Managing Large Local Data

Problem:
	•	Loading full Zillow datasets (especially post-2024) on a local 8GB Ram Mac environment consistently caused memory crashes.

Reasoning and Decision:
	•	Migrated the dataset into Google BigQuery to manage size and querying efficiently.
	•	Sliced the dataset to focus only on 2024 and beyond for quick local analysis without overwhelming local resources.

⸻

3. Imagen Image API Integration — Configuration and Key Management

Problem:
	•	No major technical barriers — primary work involved setting up the Imagen API.

Reasoning and Decision:
	•	Focused on configuring Google application credentials (application_default_credentials.json) and ensuring API access worked securely and reliably within the build environment.