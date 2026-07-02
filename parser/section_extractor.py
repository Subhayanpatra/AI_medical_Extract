def extract_sections(soup) -> dict:
    """Extract BioC passage text grouped by section type."""
    if soup is None:
        return {}

    sections = {}

    for passage in soup.find_all("passage"):
        section = "Unknown"

        for infon in passage.find_all("infon"):
            if infon.get("key") == "section_type":
                section = infon.text

        text = passage.find("text")
        if text:
            sections.setdefault(section, [])
            sections[section].append(text.get_text(" ", strip=True))

    sections.pop("REF", None)
    return sections
