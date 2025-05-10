from huggingface_hub import hf_hub_download

class HFStore:
    def download(self, repo_id, filename):
        return hf_hub_download(repo_id=repo_id, filename=filename)
