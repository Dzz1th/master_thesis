import re
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import json 

from engine.logger import make_logger

logger = make_logger(__name__)

def _process_paragraph_for_tags(paragraph_text: str, tag_highlight_mapping: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Processes a single paragraph string containing various topic-specific tags,
    and converts it into a list of dictionaries representing tagged and untagged segments.
    """
    content_pieces = []
    current_pos = 0  # Current position in the paragraph_text

    all_tags_found = []
    for tag_type_key, highlight_labels in tag_highlight_mapping.items():
        start_tag_literal = f"<{tag_type_key}_start>"
        end_tag_literal = f"<{tag_type_key}_end>"
        for match in re.finditer(rf'{re.escape(start_tag_literal)}(.*?){re.escape(end_tag_literal)}', paragraph_text, re.DOTALL):
            all_tags_found.append({
                "start_idx": match.start(),
                "end_idx": match.end(),
                "text_content": match.group(1).strip(),
                "highlights": highlight_labels
            })

    if not all_tags_found:
        stripped_paragraph = paragraph_text.strip()
        if stripped_paragraph:
            content_pieces.append({"text": stripped_paragraph, "highlights": []})
        return content_pieces

    all_tags_found.sort(key=lambda x: x["start_idx"])

    for tag_info in all_tags_found:
        if tag_info["start_idx"] > current_pos:
            untagged_segment = paragraph_text[current_pos:tag_info["start_idx"]].strip()
            # Ensure not to add just tag remnants if a tag was stripped
            # This check can be made more robust based on actual problematic remnants
            possible_tag_remnant_pattern = r"^<[/\w_]+>$|^[\w_]+>$|^<[/\w_]+$"
            if untagged_segment and not re.match(possible_tag_remnant_pattern, untagged_segment):
                content_pieces.append({"text": untagged_segment, "highlights": []})
        
        if tag_info["text_content"]:
            content_pieces.append({
                "text": tag_info["text_content"],
                "highlights": tag_info["highlights"]
            })
        current_pos = tag_info["end_idx"]

    if current_pos < len(paragraph_text):
        remaining_segment = paragraph_text[current_pos:].strip()
        possible_tag_remnant_pattern = r"^<[/\w_]+>$|^[\w_]+>$|^<[/\w_]+$"
        if remaining_segment and not re.match(possible_tag_remnant_pattern, remaining_segment):
            content_pieces.append({"text": remaining_segment, "highlights": []})
            
    return content_pieces

def _split_text_into_paragraphs(text: str) -> List[str]:
    """Splits text into paragraphs based on one or more blank lines."""
    if not text.strip():
        return []
    # Normalize line endings to \n then split by \n\n+ (one or more blank lines)
    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
    # A blank line is \n followed by optional whitespace and then \n
    # Simpler split: split by lines, then group non-empty lines.
    lines = normalized_text.split('\n')
    paragraphs = []
    current_paragraph_lines = []
    for line in lines:
        if line.strip(): # Line has content
            current_paragraph_lines.append(line)
        else: # Blank line
            if current_paragraph_lines:
                paragraphs.append("\n".join(current_paragraph_lines))
                current_paragraph_lines = []
    if current_paragraph_lines: # Add the last paragraph if any
        paragraphs.append("\n".join(current_paragraph_lines))
    return paragraphs

def parse_unified_tagged_text(
    tagged_text: str, 
    tag_highlight_mapping: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Parses text tagged by a unified LLM prompt.
    Removes specified header/footer lines and identifies Q&A sessions.
    Outputs a structured list of main content and Q&A turns.
    """
    processed_segments = []
    page_line_pattern = re.compile(
        r"^\s*[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+Chair\s+Powell's\s+Press\s+Conference\s+FINAL\s+Page\s+\d+\s+of\s+\d+\s*$", 
        re.IGNORECASE
    )
    lines = tagged_text.splitlines()
    cleaned_lines = [line for line in lines if not page_line_pattern.match(line)]
    text_to_process = "\n".join(cleaned_lines)

    speaker_pattern_regex = re.compile(r"^\s*(([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+)+)\.)", re.MULTILINE)
    first_speaker_match = speaker_pattern_regex.search(text_to_process)
    
    pre_qa_text = text_to_process
    qa_section_text = ""

    if first_speaker_match:
        split_point_index = first_speaker_match.start()
        pre_qa_text = text_to_process[:split_point_index]
        qa_section_text = text_to_process[split_point_index:]

    if pre_qa_text.strip():
        pre_qa_paragraphs = _split_text_into_paragraphs(pre_qa_text)
        for para_text in pre_qa_paragraphs:
            content_pieces = _process_paragraph_for_tags(para_text, tag_highlight_mapping)
            if content_pieces:
                 processed_segments.append({"type": "main_content", "content": content_pieces})
    
    if qa_section_text.strip():
        current_speaker_id_full = None # e.g. "STEVE LIESMAN."
        current_speaker_name_only = None # e.g. "STEVE LIESMAN"
        current_turn_text_lines = [] # Stores lines of text for the current speaker
        
        qa_lines = qa_section_text.splitlines()

        for line_text in qa_lines:
            speaker_match = speaker_pattern_regex.match(line_text)
            
            if speaker_match:
                # New speaker detected. Process previous speaker's turn.
                if current_speaker_name_only and current_turn_text_lines:
                    turn_full_text = "\n".join(current_turn_text_lines).strip()
                    turn_paragraphs = _split_text_into_paragraphs(turn_full_text)
                    turn_para_objects = []
                    for para_t in turn_paragraphs:
                        if para_t.strip(): # Ensure paragraph is not just whitespace
                            pieces = _process_paragraph_for_tags(para_t, tag_highlight_mapping)
                            if pieces:
                                turn_para_objects.append({"type": "paragraph", "content": pieces})
                    if turn_para_objects:
                        processed_segments.append({
                            "type": "qa_turn", 
                            "speaker": current_speaker_name_only, 
                            "content": turn_para_objects
                        })
                
                # Start new turn
                current_speaker_id_full = speaker_match.group(1) # Full ID with dot, e.g., "STEVE LIESMAN."
                current_speaker_name_only = speaker_match.group(2) # Name only, e.g., "STEVE LIESMAN"
                # Remove the speaker ID from the line before adding to turn text
                line_content_after_speaker = line_text[len(current_speaker_id_full):].strip()
                current_turn_text_lines = [line_content_after_speaker] if line_content_after_speaker else []
            elif current_speaker_name_only: # If already in a speaker's turn
                current_turn_text_lines.append(line_text.strip())
        
        # Add the last turn after the loop finishes
        if current_speaker_name_only and current_turn_text_lines:
            turn_full_text = "\n".join(current_turn_text_lines).strip()
            turn_paragraphs = _split_text_into_paragraphs(turn_full_text)
            turn_para_objects = []
            for para_t in turn_paragraphs:
                if para_t.strip(): # Ensure paragraph is not just whitespace
                    pieces = _process_paragraph_for_tags(para_t, tag_highlight_mapping)
                    if pieces:
                        turn_para_objects.append({"type": "paragraph", "content": pieces})
            if turn_para_objects:
                processed_segments.append({
                    "type": "qa_turn", 
                    "speaker": current_speaker_name_only, 
                    "content": turn_para_objects
                })
                
    return processed_segments

def convert_tagged_text_to_json(
    text: str, 
    tag_highlight_mapping: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Convert text with <start> and <end> tags from different tagging pipelines into a formatted JSON structure.
    
    Args:
        text: The original text with <start> and <end> tags
        tag_highlight_mapping: Dictionary mapping tag types to highlight labels
                               e.g. {"inflation": ["hawkish"], "employment": ["dovish"]}
    
    Returns:
        List of paragraph objects with formatted content and highlights
    """
    # Split the text into paragraphs
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
    
    # Process each paragraph
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        content_pieces = []
        current_pos = 0
        
        # Find all tagged sections in this paragraph
        all_tags = []
        for tag_type, highlight_labels in tag_highlight_mapping.items():
            # For each tag type, find all tagged sections
            for match in re.finditer(rf'<{tag_type}_start>(.*?)<{tag_type}_end>', paragraph, re.DOTALL):
                start_idx = match.start()
                end_idx = match.end()
                tagged_content = match.group(1)
                all_tags.append((start_idx, end_idx, tagged_content, highlight_labels))
        
        # Sort tags by their start position
        all_tags.sort(key=lambda x: x[0])
        
        # If no tags found, add the whole paragraph as unformatted text
        if not all_tags:
            content_pieces.append({
                "text": paragraph.strip(),
                "highlights": []
            })
        else:
            # Process text with tags
            for start_idx, end_idx, tagged_content, highlight_labels in all_tags:
                # Add any text before this tag
                if start_idx > current_pos:
                    untagged_text = paragraph[current_pos:start_idx].replace(
                        f"<inflation_start>", "").replace(f"<inflation_end>", ""
                    ).replace(f"<employment_start>", "").replace(f"<employment_end>", ""
                    ).replace(f"<interest_rate_start>", "").replace(f"<interest_rate_end>", ""
                    ).replace(f"<balance_sheet_start>", "").replace(f"<balance_sheet_end>", "")
                    
                    if untagged_text.strip():
                        content_pieces.append({
                            "text": untagged_text.strip(),
                            "highlights": []
                        })
                
                # Add the tagged content with its highlights
                content_pieces.append({
                    "text": tagged_content.strip(),
                    "highlights": highlight_labels
                })
                
                current_pos = end_idx
            
            # Add any remaining text after the last tag
            if current_pos < len(paragraph):
                remaining_text = paragraph[current_pos:].replace(
                    f"<inflation_start>", "").replace(f"<inflation_end>", ""
                ).replace(f"<employment_start>", "").replace(f"<employment_end>", ""
                ).replace(f"<interest_rate_start>", "").replace(f"<interest_rate_end>", ""
                ).replace(f"<balance_sheet_start>", "").replace(f"<balance_sheet_end>", "")
                
                if remaining_text.strip():
                    content_pieces.append({
                        "text": remaining_text.strip(),
                        "highlights": []
                    })
        
        # Add the processed paragraph to the result
        if content_pieces:
            formatted_paragraphs.append({
                "type": "paragraph",
                "content": content_pieces
            })
    
    return formatted_paragraphs


def convert_simple_tagged_text_to_json(text: str, highlight_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Convert text with simple <start> and <end> tags into a formatted JSON structure.
    This is a simplified version for when you're working with only one type of tag.
    
    Args:
        text: The original text with <start> and <end> tags
        highlight_labels: List of highlight labels to apply to tagged sections
                          e.g. ["hawkish", "forward_guidance"]
    
    Returns:
        List of paragraph objects with formatted content and highlights
    """
    # Split the text into paragraphs
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
    
    # Process each paragraph
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        content_pieces = []
        current_pos = 0
        
        # Find all tagged sections in this paragraph
        matches = list(re.finditer(r'<start>(.*?)<end>', paragraph, re.DOTALL))
        
        # If no tags found, add the whole paragraph as unformatted text
        if not matches:
            content_pieces.append({
                "text": paragraph.strip(),
                "highlights": []
            })
        else:
            # Process text with tags
            for match in matches:
                start_idx = match.start()
                end_idx = match.end()
                tagged_content = match.group(1)
                
                # Add any text before this tag
                if start_idx > current_pos:
                    untagged_text = paragraph[current_pos:start_idx].replace("<start>", "").replace("<end>", "")
                    if untagged_text.strip():
                        content_pieces.append({
                            "text": untagged_text.strip(),
                            "highlights": []
                        })
                
                # Add the tagged content with its highlights
                content_pieces.append({
                    "text": tagged_content.strip(),
                    "highlights": highlight_labels
                })
                
                current_pos = end_idx
            
            # Add any remaining text after the last tag
            if current_pos < len(paragraph):
                remaining_text = paragraph[current_pos:].replace("<start>", "").replace("<end>", "")
                if remaining_text.strip():
                    content_pieces.append({
                        "text": remaining_text.strip(),
                        "highlights": []
                    })
        
        # Add the processed paragraph to the result
        if content_pieces:
            formatted_paragraphs.append({
                "type": "paragraph",
                "content": content_pieces
            })
    
    return formatted_paragraphs


def merge_tagged_outputs(
    text: str,
    tagged_outputs: Dict[str, str],
    tag_highlight_mapping: Dict[str, List[str]]  # This parameter is kept for API consistency with combine_tagged_outputs_to_json
) -> str:
    """
    Merge multiple tagged outputs into a single text with differentiable tags.
    
    Args:
        text: The original untagged text
        tagged_outputs: Dictionary mapping tag type to tagged text
                        e.g. {"inflation_tagging_prompt": "<start>some text<end>"}
        tag_highlight_mapping: Dictionary mapping tag types to highlight labels (for reference only by the calling function)
    
    Returns:
        Text with differentiated tags like <inflation_tagging_prompt_start>...<inflation_tagging_prompt_end>
    """
    logger.info(f"Original text received by merge_tagged_outputs: '{text[:200]}...'") # Log beginning of text
    current_result_text = text  # Initialize with original untagged text

    # Iterate over each type of tagging (e.g., inflation, employment)
    for tag_key, individual_llm_tagged_text in tagged_outputs.items():
        logger.info(f"Processing tag_key: {tag_key}")
        logger.debug(f"  LLM tagged text for {tag_key}: '{individual_llm_tagged_text[:200]}...'")

        start_tag_to_insert = f"<{tag_key}_start>"
        end_tag_to_insert = f"<{tag_key}_end>"
        
        all_insertions_for_this_tag_key = []
        
        for match in re.finditer(r'<start>(.*?)<end>', individual_llm_tagged_text, re.DOTALL):
            content_from_llm = match.group(1).strip() # Strip whitespace
            if not content_from_llm: # Skip if content is empty after stripping
                logger.debug(f"  Skipping empty content_from_llm for {tag_key}")
                continue
            
            logger.debug(f"  Extracted content_from_llm (stripped): '{content_from_llm}'")
            
            search_offset = 0
            occurrences_found = 0
            while search_offset < len(current_result_text):
                content_idx = current_result_text.find(content_from_llm, search_offset)
                if content_idx == -1:
                    break 
                occurrences_found +=1
                logger.debug(f"    Found '{content_from_llm}' in current_result_text at index {content_idx} (occurrence {occurrences_found})")

                start_pos = content_idx
                end_pos = content_idx + len(content_from_llm)
                
                all_insertions_for_this_tag_key.append((end_pos, end_tag_to_insert))
                all_insertions_for_this_tag_key.append((start_pos, start_tag_to_insert))
                
                search_offset = end_pos 
            if occurrences_found == 0:
                logger.warning(f"  Content '{content_from_llm}' for tag_key '{tag_key}' not found in current_result_text.")


        if not all_insertions_for_this_tag_key:
            logger.info(f"  No insertions for tag_key: {tag_key}. Skipping.")
            continue

        all_insertions_for_this_tag_key.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"  Sorted insertions for {tag_key}: {all_insertions_for_this_tag_key}")
        
        temp_result_list = list(current_result_text)
        for idx, tag_string_to_insert in all_insertions_for_this_tag_key:
            temp_result_list.insert(idx, tag_string_to_insert)
        
        current_result_text = "".join(temp_result_list)
        logger.info(f"  current_result_text after {tag_key}: '{current_result_text[:200]}...'")
    
    logger.info(f"Final merged_text: '{current_result_text[:200]}...'")
    return current_result_text


def combine_tagged_outputs_to_json(
    text: str,
    tagged_outputs: Dict[str, str],
    tag_highlight_mapping: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Process multiple tagged outputs from different pipelines and convert to JSON format.
    
    Args:
        text: The original untagged text
        tagged_outputs: Dictionary mapping tag type to tagged text
                        e.g. {"inflation": "<start>some text<end>", "employment": "<start>other text<end>"}
        tag_highlight_mapping: Dictionary mapping tag types to highlight labels
                               e.g. {"inflation": ["hawkish"], "employment": ["dovish"]}
    
    Returns:
        Formatted JSON structure with paragraphs and highlights
    """
    # First merge all the tagged outputs with different tag markers
    merged_text = merge_tagged_outputs(text, tagged_outputs, tag_highlight_mapping)
    
    # Then convert to JSON
    return convert_tagged_text_to_json(merged_text, tag_highlight_mapping)


def _clean_text(text: str) -> str:
    """
    Cleans the text by removing header lines and page numbers.
    
    Args:
        text: The original text to be cleaned
        
    Returns:
        Cleaned text with headers and page numbers removed
    """
    lines = text.splitlines()
    cleaned_lines = []
    
    # Patterns to match and remove
    header_pattern = re.compile(r'^.*\d{1,2},\s+\d{4}.*Press\s+Conference\s+FINAL.*$', re.IGNORECASE)
    page_pattern = re.compile(r'^.*Page\s+\d+\s+of\s+\d+.*$', re.IGNORECASE)
    
    for line in lines:
        # Skip lines matching header or page patterns
        if header_pattern.match(line) or page_pattern.match(line):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def generate_qa_chunks(text: str) -> List[Dict[str, Any]]:
    """
    Splits text into chunks where first chunk is the introductory statement,
    and subsequent chunks group questions and answers by journalist.
    Excludes Federal Reserve Chair names from being treated as speakers only in Q&A section.
    Merges consecutive chunks from the same speaker.
    Preserves reporter names in content for multiple questions from the same reporter.
    Filters out chunks with content shorter than 100 characters.
    
    Args:
        text: The original text to be processed
        
    Returns:
        List of chunks, where each chunk is a dictionary containing:
        - type: "intro" for introductory statement or "qa_group" for Q&A
        - speaker: Name of the journalist (for Q&A chunks)
        - content: String containing the chunk's content
    """
    # Clean the text first
    text = _clean_text(text)
    
    chunks = []
    
    # List of Federal Reserve Chair names to exclude from Q&A section
    chair_names = [
        'CHAIRMAN BERNANKE',
        'CHAIR BERNANKE',
        'CHAIR YELLEN',
        'CHAIR POWELL'
    ]
    
    # Split text into pre-QA and QA sections
    # Look for the first journalist question, not any speaker
    journalist_pattern = re.compile(r"^\s*(([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+)+)\.)", re.MULTILINE)
    first_journalist_match = None
    
    # Find the first journalist (non-chair) speaker
    for match in journalist_pattern.finditer(text):
        speaker_name = match.group(2)
        if speaker_name not in chair_names:
            first_journalist_match = match
            break
    
    pre_qa_text = text
    qa_section_text = ""
    
    if first_journalist_match:
        split_point_index = first_journalist_match.start()
        pre_qa_text = text[:split_point_index]
        qa_section_text = text[split_point_index:]
    
    # Process introductory statement - include all content including chair statements
    if pre_qa_text.strip():
        intro_paragraphs = _split_text_into_paragraphs(pre_qa_text)
        if intro_paragraphs:
            # Filter out empty paragraphs
            intro_paragraphs = [p for p in intro_paragraphs if p.strip()]
            if intro_paragraphs:  # Check again after filtering
                intro_content = "\n".join(intro_paragraphs)
                if len(intro_content) >= 100:
                    chunks.append({
                        "type": "intro",
                        "content": intro_content
                    })
    
    # Process Q&A section - here we exclude chair statements
    if qa_section_text.strip():
        current_speaker = None
        current_chunk_lines = []
        question_count = 0
        
        for line in qa_section_text.splitlines():
            speaker_match = journalist_pattern.match(line)
            
            if speaker_match:
                speaker_name = speaker_match.group(2)  # Name without the dot
                
                # Skip if the speaker is a Federal Reserve Chair
                if speaker_name in chair_names:
                    if current_speaker:
                        current_chunk_lines.append(line.strip())
                    continue
                
                # If we have a previous speaker's content and it's a different speaker, save it as a chunk
                if current_speaker and current_speaker != speaker_name and current_chunk_lines:
                    chunk_content = "\n".join(current_chunk_lines)
                    if len(chunk_content) >= 100:
                        chunks.append({
                            "type": "qa_group",
                            "speaker": current_speaker,
                            "content": chunk_content
                        })
                    current_chunk_lines = []
                    question_count = 0
                
                # Start or continue chunk for current speaker
                current_speaker = speaker_name
                line_content = line[len(speaker_match.group(1)):].strip()
                
                # If this is a new question from the same reporter, increment question count
                if current_speaker == speaker_name and line_content:
                    question_count += 1
                
                # If this is a follow-up question (question_count > 1), include the reporter's name
                if question_count > 1 and line_content:
                    current_chunk_lines.append(f"{speaker_name}. {line_content}")
                elif line_content:
                    current_chunk_lines.append(line_content)
            elif current_speaker:
                current_chunk_lines.append(line.strip())
        
        # Add the last chunk if there is one
        if current_speaker and current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if len(chunk_content) >= 100:
                chunks.append({
                    "type": "qa_group",
                    "speaker": current_speaker,
                    "content": chunk_content
                })
    
    return chunks


# Example usage
if __name__ == "__main__":
    # Load the LLM output
    # with open("results.json", "r") as f:
    #     loaded_data = json.load(f)
    # unified_llm_output_text = loaded_data['result']

    # # Define the mapping for the unified tags.
    # # Keys should be the base topic names used in your unified prompt's tags
    # # (e.g., "employment" for "<employment_start>").
    # tag_mapping_for_parser = {
    #     "employment": ["employment"], # Choose your highlight labels/styles
    #     "inflation": ["inflation"],
    #     "interest_rate": ["interest rate"],
    #     "balance_sheet": ["balance sheet"]
    #     # Add any other topics your unified prompt handles
    # }

    # parsed_structured_data = parse_unified_tagged_text(unified_llm_output_text, tag_mapping_for_parser)

    # # Save the new structured output
    # with open("parsed_results.json", "w") as f:
    #     json.dump(parsed_structured_data, f, indent=2)
        
    # print("Parsed data using unified parser saved to parsed_results.json")

    df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/pc_data/summarized_data.csv")

    text = df['text'].iloc[10]
    

    chunks = generate_qa_chunks(text)

    print(chunks)


