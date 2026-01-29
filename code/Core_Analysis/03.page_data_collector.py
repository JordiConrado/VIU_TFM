import networkx as nx
import requests
import time
import pandas as pd
import os


class WikipediaRepository:
    def __init__(self, repo_path="wikipedia_page_metadata.csv", language_code="es"):
        """
        Initializes the repository.
        :param repo_path: Path to the CSV file acting as the local database.
        :param language_code: Default is 'es' for Spanish Wikipedia queries.
        """
        self.repo_path = repo_path
        self.api_url = f"https://{language_code}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        # Set a user-agent as per Wikimedia API policy
        self.session.headers.update({
            'User-Agent': 'MetadataExtractorBot/1.0 (contact: your-email@example.com)'
        })
        self.existing_data = self._load_existing_repo()

    def _load_existing_repo(self):
        """Loads the current repository to avoid redundant API calls."""
        if os.path.exists(self.repo_path):
            try:
                # Treat page_id as string to avoid floating point issues with large IDs
                df = pd.read_csv(self.repo_path, dtype={'page_id': str})
                print(f"Repository loaded: {len(df)} records found.")
                return df
            except Exception as e:
                print(f"Warning: Could not read repository, starting fresh. Error: {e}")
        return pd.DataFrame(columns=["page_id", "page_title", "categories", "url"])

    def fetch_new_metadata(self, identifiers, is_id=True):
        """Fetches metadata for IDs not currently present in the local CSV."""
        ids_to_check = set(map(str, identifiers))
        existing_ids = set(self.existing_data['page_id'].astype(str))

        new_ids = list(ids_to_check - existing_ids)

        if not new_ids:
            print("No new pages to fetch. Repository is up to date.")
            return []

        print(f"Fetching metadata for {len(new_ids)} new pages from Spanish Wikipedia ({self.api_url})...")
        results = []
        batch_size = 50

        for i in range(0, len(new_ids), batch_size):
            batch = new_ids[i:i + batch_size]
            param_key = "pageids" if is_id else "titles"

            params = {
                "action": "query",
                "format": "json",
                "prop": "categories|info",
                "inprop": "url",
                param_key: "|".join(batch),
                "cllimit": "max",
                "clshow": "!hidden"  # Hide administrative/hidden categories
            }

            retries = 0
            while retries < 5:
                try:
                    response = self.session.get(self.api_url, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()

                    pages = data.get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        if "missing" in pdata or pid == "-1":
                            continue

                        # Clean the 'Categoría:' or 'Category:' prefix
                        categories = [c['title'].split(':', 1)[-1] for c in pdata.get('categories', [])]

                        results.append({
                            "page_id": str(pid),
                            "page_title": pdata.get("title"),
                            "categories": "|".join(categories),
                            "url": pdata.get("fullurl")
                        })
                    break
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Request failed. Retrying in {wait_time}s... ({retries}/5)")
                    time.sleep(wait_time)

        return results

    def update_repository(self, graph_files, node_is_id=True):
        """Extracts node IDs from graph files and appends missing metadata to the CSV."""
        all_node_ids = set()
        for file in graph_files:
            if os.path.exists(file):
                try:
                    G = nx.read_graphml(file)
                    all_node_ids.update(list(G.nodes()))
                except Exception as e:
                    print(f"Error reading {file}: {e}")

        if not all_node_ids:
            print("No nodes found in the provided graph files.")
            return

        # Fetch only what we don't already have locally
        new_records = self.fetch_new_metadata(list(all_node_ids), is_id=node_is_id)

        if new_records:
            new_df = pd.DataFrame(new_records)
            # Combine existing data with new findings and save
            updated_df = pd.concat([self.existing_data, new_df], ignore_index=True)
            updated_df.to_csv(self.repo_path, index=False, encoding='utf-8-sig')
            self.existing_data = updated_df
            print(f"Repository successfully updated. Total records: {len(updated_df)}")
        else:
            print("All nodes from graph files already exist in the local repository.")


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

    PAGE_METADATA_FILE_PATH = f'{CSV_DIR}wikipedia_page_metadata.csv'
    file1 = f'{GRAPH_DIR}largest_connected_component_2014_02_20251221_160205.graphml'
    file2 = f'{GRAPH_DIR}largest_connected_component_2014_03_20251221_160211.graphml'

    graph_list = [file1, file2]
    repo = WikipediaRepository(repo_path=PAGE_METADATA_FILE_PATH, language_code="es")
    repo.update_repository(graph_list)