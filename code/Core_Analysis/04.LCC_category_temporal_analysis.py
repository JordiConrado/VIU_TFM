import pandas as pd
import networkx as nx
from collections import Counter
import time
import os
import requests

"""
Analysis of category distribution within the Largest Connected Component (LCC)
of a bipartite graph representing user-page interactions over time.
Categories are derived from a provided mapping of page IDs to categories.

Change log:
    2025-12-21: Initial implementation of LCC category analysis.
"""


class CategoryMapper:
    """
    Handles fetching and local storage of Wikipedia pages using Page IDs
    to resolve associated categories and their internal Category IDs.
    """

    def __init__(self, cache_filepath="wikipedia_category_map.csv"):
        self.cache_file = cache_filepath
        self.api_url = "https://es.wikipedia.org/w/api.php"
        self.headers = {'User-Agent': 'CategoryMapper/1.0 (your_email@example.com)'}
        self.cache_df = self._load_cache()

    def _load_cache(self):
        """Loads the existing CSV or creates a new one with required headers."""
        if os.path.exists(self.cache_file):
            return pd.read_csv(self.cache_file)
        return pd.DataFrame(columns=['page_id', 'page_title', 'category_id', 'category_title'])

    def _save_cache(self):
        """Saves data to the CSV file."""
        self.cache_df.to_csv(self.cache_file, index=False)

    def _request_with_retry(self, params, retries=5):
        """Implements exponential backoff for API requests."""
        delay = 1
        for i in range(retries):
            try:
                response = requests.get(self.api_url, params=params, headers=self.headers)
                response.raise_for_status()
                return response.json()
            except Exception:
                if i == retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    def fetch_categories_by_id(self, page_ids):
        """
        Fetches category titles and their internal IDs using a list of Page IDs.
        Using IDs instead of titles prevents issues with page redirects or renames.
        """
        new_records = []
        # Convert IDs to strings for the join operation
        str_page_ids = [str(pid) for pid in page_ids]

        # Process in batches of 50
        for i in range(0, len(str_page_ids), 50):
            batch = str_page_ids[i:i + 50]

            # Step 1: Get categories for the page IDs
            params = {
                "action": "query",
                "format": "json",
                "prop": "categories",
                "pageids": "|".join(batch),
                "cllimit": "max"
            }

            data = self._request_with_retry(params)
            if not data:
                continue

            pages_data = data.get("query", {}).get("pages", {})

            # Collect all category titles to resolve their unique IDs
            all_category_full_titles = []
            for page_info in pages_data.values():
                if "categories" in page_info:
                    all_category_full_titles.extend([c["title"] for c in page_info["categories"]])

            # Step 2: Resolve category_id for the category titles
            category_id_map = {}
            if all_category_full_titles:
                unique_cats = list(set(all_category_full_titles))
                for j in range(0, len(unique_cats), 50):
                    cat_batch = unique_cats[j:j + 50]
                    id_params = {
                        "action": "query",
                        "format": "json",
                        "titles": "|".join(cat_batch)
                    }
                    id_data = self._request_with_retry(id_params)
                    if id_data:
                        cat_pages = id_data.get("query", {}).get("pages", {})
                        for cid, cval in cat_pages.items():
                            category_id_map[cval["title"]] = cid

            # Step 3: Map page IDs to category IDs
            for pid, val in pages_data.items():
                p_title = val.get("title")
                if "categories" in val:
                    for cat in val["categories"]:
                        cat_full_title = cat["title"]
                        cat_id = category_id_map.get(cat_full_title)
                        clean_cat_title = cat_full_title.replace("Categoría:", "")

                        new_records.append({
                            'page_id': pid,
                            'page_title': p_title,
                            'category_id': cat_id,
                            'category_title': clean_cat_title
                        })

        if new_records:
            new_df = pd.DataFrame(new_records)
            self.cache_df = pd.concat([self.cache_df, new_df], ignore_index=True)
            # Ensure we don't duplicate the same page-to-category mapping
            self.cache_df.drop_duplicates(subset=['page_id', 'category_id'], inplace=True)
            self._save_cache()
            print(f"Update complete. {len(new_records)} mappings processed.")


def analyze_bipartite_lcc_with_cache(B_lcc, cache_path):
    """
    Integration function for your workflow.
    """
    # 1. Identify page nodes in the Bipartite LCC
    page_nodes = [n for n, d in B_lcc.nodes(data=True) if d.get('bipartite') == 1]

    # 2. Initialize/Update Cache
    manager = CategoryMapper(cache_path)
    manager.fetch_categories_by_id(page_nodes)

    # 3. Perform Category Analysis
    stats = []
    for p in page_nodes:
        cats = category_map.get(p, ["Uncategorized"])
        for c in cats:
            stats.append({"page": p, "category": c})

    return pd.DataFrame(stats)


if __name__ == '__main__':
    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

    START_YEAR = 2012
    END_YEAR = 2014
    RUN = '14'

    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'

    b_subgraph_file = f'{GRAPH_DIR}b_subgraph_lcc_2014_04_20251221_174817.graphml'
    cache_path = f'{CSV_DIR}wikipedia_category_cache.csv'

    b_lcc = nx.read_graphml(b_subgraph_file)
    df_cats = analyze_bipartite_lcc_with_cache(b_lcc, cache_path)
    print(df_cats['category'].value_counts().head(10))

    print(f"\nEnd time [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
