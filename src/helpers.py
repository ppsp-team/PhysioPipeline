from pathlib import Path


def extract_raw_pathname(coi_structure: list, FAMILY_ID: int, SEANCE_ID: int, ROLE_ID: int) -> Path:
    """
    Extracts the raw sensor and session code from the COI structure based on family and session IDs.

    Args:
        coi_structure (list): The COI structure containing sensor and session information.
        FAMILY_ID (int): The family ID to filter the entries.
        SEANCE_ID (int): The session ID to filter the entries.
        ROLE_ID (int): The role ID to filter the entries.

    Returns:
        

    Raises:
        ValueError: If no matching entry is found in the COI structure.
    """
    sensor, session_code,  = [
        (entry['sensor'], entry['session_code'])
        for entry in coi_structure
        if entry.get('family')  == FAMILY_ID
        and entry.get('session') == SEANCE_ID
        and entry.get('index')   == ROLE_ID
    ][0]

    if not sensor or not session_code:
        raise ValueError(f"No matching entry found for family ID {FAMILY_ID}, session ID {SEANCE_ID}, and role ID {ROLE_ID}.")
    
    return Path(f"../data/raw/segmented_physio/{session_code}/{sensor}_segmented.xlsx")
