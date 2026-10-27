
from shared_retriever import get_retrievers

queries = [
    "Domiciliary Treatment: treatment at home exceeding three days, doctor's certification, patient cannot be moved to hospital",
    "Policy coverage for home treatment when a patient cannot be moved to a hospital",
    "Section C.6 Domiciliary Treatment",
]

retrievers = get_retrievers(k=10)

strategies = {
    "BM25": retrievers["bm25"],
    "Dense": retrievers["dense"],
    "RRF Hybrid": retrievers["hybrid_rrf"],
}

for i, query in enumerate(queries, start=1):
    print(f"\n{'=' * 70}")
    print(f"QUERY {i}: {query}")
    print("=" * 70)

    for name, retrieve in strategies.items():
        print(f"\n--- {name} ---")

        try:
            results = retrieve(query)

            found_c6 = False

            for rank, doc in enumerate(results, start=1):
                clause_id = doc.metadata.get("id", "Unknown")
                if clause_id == "C.6":
                    found_c6 = True

                print(f"{rank}. Clause: {clause_id}")
                print(f"   Text: {doc.page_content[:250]}")

            print(f"C.6 FOUND: {found_c6}")

        except Exception as exc:
            print(f"ERROR: {exc}")
