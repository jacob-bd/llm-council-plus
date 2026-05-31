"""JSON-based storage for conversations."""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR


INDEX_FILE_NAME = "conversations_index.json"


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def get_index_path() -> str:
    """Get the file path for the conversation index."""
    return os.path.join(DATA_DIR, INDEX_FILE_NAME)


def _load_index() -> Optional[List[Dict[str, Any]]]:
    """Load the conversation index file."""
    path = get_index_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_index(index: List[Dict[str, Any]]):
    """Save the conversation index file."""
    ensure_data_dir()
    path = get_index_path()
    with open(path, 'w') as f:
        json.dump(index, f, indent=2)


def rebuild_index() -> List[Dict[str, Any]]:
    """
    Rebuild the conversation index from actual conversation files.
    Use this fallback if index is missing or corrupted.
    """
    ensure_data_dir()
    index = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json') and filename != INDEX_FILE_NAME:
            path = os.path.join(DATA_DIR, filename)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    index.append({
                        "id": data["id"],
                        "created_at": data["created_at"],
                        "title": data.get("title", "New Conversation"),
                        "mode": data.get("mode", "council"),
                        "message_count": len(data["messages"])
                    })
            except (json.JSONDecodeError, OSError):
                continue

    # Sort by creation time, newest first
    index.sort(key=lambda x: x["created_at"], reverse=True)
    _save_index(index)
    return index


def _update_index_entry(conversation: Dict[str, Any]):
    """Update or add a single entry in the index."""
    index = _load_index()
    if index is None:
        index = rebuild_index()
        return  # rebuild already includes the current state if file was saved

    # Create metadata entry
    entry = {
        "id": conversation["id"],
        "created_at": conversation["created_at"],
        "title": conversation.get("title", "New Conversation"),
        "mode": conversation.get("mode", "council"),
        "message_count": len(conversation["messages"])
    }

    # Remove existing entry if present
    index = [item for item in index if item["id"] != conversation["id"]]
    
    # Add new entry
    index.append(entry)
    
    # Sort and save
    index.sort(key=lambda x: x["created_at"], reverse=True)
    _save_index(index)


def _remove_from_index(conversation_id: str):
    """Remove an entry from the index."""
    index = _load_index()
    if index is None:
        return  # No index to remove from

    # Filter out the deleted conversation
    new_index = [item for item in index if item["id"] != conversation_id]
    
    if len(new_index) != len(index):
        _save_index(new_index)


def create_conversation(conversation_id: str, mode: str = "council") -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation
        mode: Conversation mode — "council" or "advisors"

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": "New Conversation",
        "mode": mode,
        "messages": []
    }

    # Save to file
    path = get_conversation_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)

    # Update index
    _update_index_entry(conversation)

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    path = get_conversation_path(conversation['id'])
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)

    # Update index
    _update_index_entry(conversation)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).
    Uses cached index file for O(1) performance.

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    # Try to load from index first
    index = _load_index()
    
    # If index missing or invalid, rebuild it
    if index is None:
        return rebuild_index()
        
    return index


def add_user_message(conversation_id: str, content: str, conversation: Optional[Dict[str, Any]] = None):
    """Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
        conversation: Pre-loaded conversation dict (avoids redundant disk read)
    """
    if conversation is None:
        conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    # If this is the very first message in a reused empty draft, reset the creation date to now
    if len(conversation["messages"]) == 0:
        conversation["created_at"] = datetime.now(timezone.utc).isoformat()

    conversation["messages"].append({
        "role": "user",
        "content": content
    })

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: Optional[List[Dict[str, Any]]] = None,
    stage3: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conversation: Optional[Dict[str, Any]] = None
):
    """Add an assistant message to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses (always present)
        stage2: List of model rankings (None if execution_mode was 'chat_only')
        stage3: Final synthesized response (None if execution_mode was not 'full')
        metadata: Optional metadata including execution_mode, label_to_model, etc.
        conversation: Pre-loaded conversation dict (avoids redundant disk read)
    """
    if conversation is None:
        conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {
        "role": "assistant",
        "stage1": stage1,
    }

    if stage2 is not None:
        message["stage2"] = stage2
    if stage3 is not None:
        message["stage3"] = stage3
    if metadata:
        message["metadata"] = metadata

    conversation["messages"].append(message)
    save_conversation(conversation)


def add_advisor_message(
    conversation_id: str,
    rounds: List[Dict[str, Any]],
    verdict: Optional[Dict[str, Any]] = None,
    tiebreaker: Optional[Dict[str, Any]] = None,
    personas: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conversation: Optional[Dict[str, Any]] = None
):
    """Add an advisor debate message to a conversation.

    Args:
        conversation_id: Conversation identifier
        rounds: List of round dicts, each with round_number and responses
        verdict: Structured verdict (summary, consensus, disagreements, etc.)
        tiebreaker: Tiebreaker result if vote was tied
        metadata: Optional metadata (persona_ids, models, etc.)
        conversation: Pre-loaded conversation dict (avoids redundant disk read)
    """
    if conversation is None:
        conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {
        "role": "assistant",
        "mode": "advisors",
        "rounds": rounds,
    }

    if personas is not None:
        message["personas"] = personas
    if verdict is not None:
        message["verdict"] = verdict
    if tiebreaker is not None:
        message["tiebreaker"] = tiebreaker
    if metadata:
        message["metadata"] = metadata

    conversation["messages"].append(message)
    save_conversation(conversation)


def add_error_message(conversation_id: str, error_text: str):
    """
    Add an error message to a conversation to record a failed turn.

    Args:
        conversation_id: Conversation identifier
        error_text: The error description
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {
        "role": "assistant",
        "content": None,
        "error": error_text,
        "stage1": [],
        "stage2": None,
        "stage3": None
    }

    conversation["messages"].append(message)
    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if deleted, False if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return False

    os.remove(path)
    
    # Update index
    _remove_from_index(conversation_id)
    
    return True
