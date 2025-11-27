from git import Repo


def get_git_commit_hash() -> str:
    try:
        repo = Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
    except Exception:
        commit_hash = "Unknown"
    return commit_hash
