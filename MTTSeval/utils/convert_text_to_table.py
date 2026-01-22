import re
import os
import sys
import math
import argparse
import collections
import numpy as np
import pandas as pd
from itertools import groupby
from tqdm import tqdm
from transformers import AutoTokenizer
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process
from multiprocessing import Pool, cpu_count

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.configs import get_config


# Utility functions for token operations
class TokenUtils:
    @staticmethod
    def decode_tokens(tokenizer, tokens):
        """Decode a list of tokens using the tokenizer."""
        return tokenizer.decode(tokens)

    @staticmethod
    def encode_text(tokenizer, text):
        """Encode a string of text to tokens (excluding special tokens)."""
        return tokenizer.encode(text)[1:-1]

    @staticmethod
    def find_closest_match(word, possible_list):
        """Find the closest match for a word in a list based on similarity."""
        possible_list = list(possible_list.keys())
        match = process.extractOne(word, possible_list, scorer=Levenshtein.normalized_distance)
        return match[0]

    @staticmethod
    def clean_numeric_content(value):
        """Clean a numeric value by removing letters and fixing common formatting issues."""
        value = value.replace("..", ".")  # Fix invalid decimal points
        # value = TokenUtils.remove_after_second_dot(value)
        value = TokenUtils.remove_letters(value)  # Remove any alphabetical characters
        return value.strip()

    @staticmethod
    def remove_letters(input_string):
        """Remove all letters from the input string."""
        return re.sub(r'[a-zA-Z]', '', input_string).strip()

    @staticmethod
    def remove_after_second_dot(value):
        parts = value.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:2])  # Keep only the content before the second dot
        return value  # Return as is if there is one or no dot

    @staticmethod
    def is_number(s):
        """Check if the string s can be converted to a float."""
        try:
            return float(s) is not None
        except ValueError:
            return False

    @staticmethod
    def restore_text(cell, compact_decimal_notation=True):
        if isinstance(cell, str):
            result = re.sub(r'(?<=\d) +(?=\d)', '', cell)
            result = re.sub(r' +(?=\.)', '', result)
            result = re.sub(r'(?<=-)\s+', '', result)
            if compact_decimal_notation:
                result = re.sub(r'\. +(?=\d)', '.', result)
            return result
        else:
            return cell
    
    @staticmethod
    def normalize_categorical_value(value, valid_vocab=None):
        """
        Normalize categorical value to match real data format.
        Handles case, spacing, and format differences.
        """
        if not isinstance(value, str):
            # Convert to string, handling float/int edge cases
            if isinstance(value, float) and value.is_integer():
                value = str(int(value))
            else:
                value = str(value)
        
        # Remove extra spaces
        value = ' '.join(value.split())
        
        # If valid_vocab is provided, try to find closest match
        if valid_vocab is not None:
            valid_vocab_list = list(valid_vocab) if not isinstance(valid_vocab, list) else valid_vocab
            # Try exact match first (case-insensitive)
            for v in valid_vocab_list:
                if str(v).lower() == value.lower():
                    return str(v)
            
            # Try fuzzy match (more lenient for case/spacing differences)
            try:
                match = process.extractOne(value, valid_vocab_list, scorer=Levenshtein.normalized_distance)
                if match and match[1] <= 0.3:  # Allow for case/spacing differences
                    return str(match[0])
            except:
                pass
        
        return value

# Main class for evaluating samples
class TableEvaluator:
    def __init__(self, config):
        self.config = config
        self.ehr = config["ehr"]    
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_event_size = config["max_event_size"]
        self.max_len = config["max_event_token_len"]
        self.pid_col = config["pid_column"]
        self.time_col = config["time_column"]
        
        self.p_token = {"table": 1, "column": 2, "content": 3}
        # Load col_type first (needed for vocabulary generation)
        self.col_type = pd.read_pickle(os.path.join(self.config["real_data_root"], self.config["col_type"]))
        
        # Load predefined vocabulary
        predef_vocab_path = os.path.join(self.config["real_data_root"], self.config["predef_vocab"])
        if os.path.exists(predef_vocab_path):
            self.p_vocab = pd.read_pickle(predef_vocab_path)
        else:
            # Generate vocabulary from real data if file doesn't exist
            print(f"Warning: {predef_vocab_path} not found. Generating vocabulary from real data...")
            self.p_vocab = self._generate_vocab_from_real_data()
        
        mapping_func_path = os.path.join(self.config["real_data_root"], f"{self.config['ehr']}_id2word.pkl")
        if os.path.exists(mapping_func_path):
            self.mapping_func = pd.read_pickle(mapping_func_path)
        else:
            print(f"Warning: {mapping_func_path} not found. Using empty mapping.")
            self.mapping_func = {}
        
        # Recovery options
        self.recovery = config["recovery"]
        self.recovery_save = config["recovery_save"]
        if self.recovery_save:
            assert self.recovery, "Recovery must be enabled if recovery_save is set to True"
        
        # Build itemid text-to-ID mapping for tables that need it
        self.itemid_mapping = self._build_itemid_mapping()

    def _generate_vocab_from_real_data(self):
        """
        Generate vocabulary from real data CSV files.
        Returns a dictionary with structure: {table_name: {column_name: [any_value, is_numeric]}}
        """
        vocab = {}
        table_names = self.config.get("table_names", ["labevents", "inputevents", "prescriptions"])
        
        for table_name in table_names:
            csv_path = os.path.join(self.config["real_data_root"], f"{table_name}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping {table_name}")
                continue
            
            # Only read column names (nrows=0 is much faster than reading data)
            df_header = pd.read_csv(csv_path, nrows=0)
            vocab[table_name] = {}
            
            # Get column types from col_type (more reliable than inferring from data)
            numeric_cols = set()
            if table_name in self.col_type:
                numeric_cols = set(self.col_type[table_name].get("numeric_columns", []))
            
            for col in df_header.columns:
                # Skip ID and time columns
                if col in [self.pid_col, self.time_col]:
                    continue
                
                # Determine if column is numeric (use col_type as primary source)
                is_numeric = col in numeric_cols
                vocab[table_name][col] = [None, is_numeric]
        
        print(f"Generated vocabulary for {len(vocab)} tables")
        return vocab

    def _build_itemid_mapping(self):
        """
        Build mapping from itemid text descriptions to numeric IDs.
        Only needed for labevents and inputevents tables.
        Returns: {table_name: {text_label: itemid}}
        """
        mapping = {}
        
        # Map for labevents: use d_labitems.csv
        # Try to find raw data root from real_data_root (go up 2 levels from processed_12)
        raw_data_root = self.config.get("raw_data_root", "")
        if not raw_data_root:
            # Infer from real_data_root: if real_data_root is data/processed_12, raw is data/raw
            real_data_root = self.config.get("real_data_root", "")
            if real_data_root:
                # Try common patterns: data/processed_12 -> data/raw
                if "processed" in real_data_root:
                    raw_data_root = real_data_root.replace("processed", "raw")
                elif "processed_12" in real_data_root:
                    raw_data_root = real_data_root.replace("processed_12", "raw")
                else:
                    # Try going up one level and then to raw
                    parent_dir = os.path.dirname(real_data_root)
                    raw_data_root = os.path.join(parent_dir, "raw")
        
        d_labitems_path = os.path.join(raw_data_root, "MIMIC-IV", "hosp", "d_labitems.csv")
        if os.path.exists(d_labitems_path):
            try:
                d_labitems = pd.read_csv(d_labitems_path, low_memory=False)
                if "itemid" in d_labitems.columns and "label" in d_labitems.columns:
                    # Create mapping: label -> itemid
                    # Handle duplicate labels by taking the first occurrence
                    lab_mapping = {}
                    for _, row in d_labitems.iterrows():
                        label = str(row["label"]).strip()
                        itemid = row["itemid"]
                        # Only add if label is not empty and not already mapped
                        if label and label != "nan" and label not in lab_mapping:
                            lab_mapping[label] = int(itemid)
                    mapping["labevents"] = lab_mapping
                    print(f"Built itemid mapping for labevents: {len(lab_mapping)} entries")
            except Exception as e:
                print(f"Warning: Failed to build labevents itemid mapping: {e}")
        
        # Map for inputevents: use icu/d_items.csv
        d_items_path = os.path.join(raw_data_root, "MIMIC-IV", "icu", "d_items.csv")
        if os.path.exists(d_items_path):
            try:
                d_items = pd.read_csv(d_items_path, low_memory=False)
                if "itemid" in d_items.columns and "label" in d_items.columns:
                    # Create mapping: label -> itemid
                    # Handle duplicate labels by taking the first occurrence
                    input_mapping = {}
                    for _, row in d_items.iterrows():
                        label = str(row["label"]).strip()
                        itemid = row["itemid"]
                        # Only add if label is not empty and not already mapped
                        if label and label != "nan" and label not in input_mapping:
                            input_mapping[label] = int(itemid)
                    mapping["inputevents"] = input_mapping
                    print(f"Built itemid mapping for inputevents: {len(input_mapping)} entries")
            except Exception as e:
                print(f"Warning: Failed to build inputevents itemid mapping: {e}")
        
        return mapping

    def _normalize_text_for_matching(self, text):
        """
        Normalize text for matching: lowercase, remove punctuation, normalize spaces.
        """
        import re
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = ' '.join(text.split())  # Normalize spaces
        return text
    
    def _map_itemid_text_to_id(self, table_name, text_value):
        """
        Map itemid text description to numeric ID using hierarchical matching strategy:
        1. Exact match (after trimming)
        2. Case-insensitive match
        3. Normalized match (remove punctuation, normalize spaces)
        4. Fuzzy match (only for longer strings, strict threshold)
        
        Returns the numeric ID if mapping exists, otherwise returns None.
        """
        if table_name not in self.itemid_mapping:
            return None
        
        mapping = self.itemid_mapping[table_name]
        text_value = str(text_value).strip()
        
        # Step 1: Exact match
        if text_value in mapping:
            return mapping[text_value]
        
        # Step 2: Case-insensitive match (most common case, ~87%)
        text_lower = text_value.lower()
        for label, itemid in mapping.items():
            if label.lower() == text_lower:
                return itemid
        
        # Step 3: Normalized match (remove punctuation, normalize spaces)
        text_normalized = self._normalize_text_for_matching(text_value)
        for label, itemid in mapping.items():
            if self._normalize_text_for_matching(label) == text_normalized:
                return itemid
        
        # Step 4: Fuzzy match (only for strings longer than 3 chars, strict threshold)
        # Avoid fuzzy matching for very short strings to prevent false matches
        if len(text_value) > 3:
            try:
                from rapidfuzz import process
                match = process.extractOne(text_value, list(mapping.keys()), scorer=Levenshtein.normalized_distance)
                if match and match[1] <= 0.2:  # Very strict: 80% similarity required
                    return mapping[match[0]]
            except:
                pass
        
        return None

    def load_samples(self):
        """Load and clean input samples."""
        if self.config["syn_data_root"] == self.config["real_data_root"]:
            input_data = np.load(os.path.join(self.config["syn_data_root"], self.config["input_file_name"]))
            type_data = np.load(os.path.join(self.config["syn_data_root"], self.config["type_file_name"]))
            time_data = np.load(os.path.join(self.config["syn_data_root"], self.config["real_time_file_name"]), allow_pickle=True)
        else:
            input_data = np.load(os.path.join(self.config["syn_data_root"], self.config["input_file_name"]))
            type_data = np.load(os.path.join(self.config["syn_data_root"], self.config["type_file_name"]))
            time_data = np.load(os.path.join(self.config["syn_data_root"], self.config["time_file_name"]))
            time_data = time_data.squeeze()
            
        time_samples = self.clean_time_sample(time_data, mode=self.config["process_time"])
        
        samples = list(zip(input_data, type_data))
        cleaned_samples = [self.clean_sample(sample) for sample in tqdm(samples)]
        input_samples, type_samples = zip(*cleaned_samples)
        return list(input_samples), list(type_samples), time_samples

    def clean_sample(self, sample):
        """Process a single sample to remove invalid events."""
        i_sample, t_sample = sample
        input_sample, type_sample = [], []

        for i_event, t_event in zip(i_sample, t_sample):
            if i_event[0] == 0:  # End of valid events
                break
            sep_idx = self.get_separator_index(i_event, t_event)
            if sep_idx is None or sep_idx == 0:
                continue  # Skip invalid events
            input_sample.append(i_event[:sep_idx])
            type_sample.append(t_event[:sep_idx])
        return input_sample, type_sample

    def get_separator_index(self, i_event, t_event):
        """Determine the separator index for an event."""
        i_sep_idx = np.where(i_event == 102)[0]
        t_sep_idx = np.where(t_event == 6)[0]
        if i_sep_idx.size and t_sep_idx.size:
            return min(i_sep_idx[0], t_sep_idx[0])
        return i_sep_idx[0] if i_sep_idx.size else t_sep_idx[0] if t_sep_idx.size else None
    
    def clean_time_sample(self, time, mode):
        """
        Preprocess time samples to fix or filter non-monotonic sequences and outliers.
        """
        obs_size = self.config["obs_size"]
        
        # Normalize the value of 'time' by rounding it down to the nearest multiple of 10
        # Example: 37 -> 30, 45 -> 40
        time = time // 10 * 10

        def fix_non_monotonic_sequence(seq):
            for idx in range(len(seq) - 1):
                seq[idx + 1] = max(seq[idx], seq[idx + 1])
            return seq

        def filter_or_fix_sequence(seq, obs_size, mode):
            # Filter values exceeding observation size
            seq = seq[:np.where(seq > obs_size * 60)[0][0]] if np.any(seq > obs_size * 60) else seq
            # Filter negative values
            seq = seq[:np.where(seq < 0)[0][0]] if np.any(seq < 0) else seq

            # Fix non-monotonic sequences
            non_monotonic_index = np.where(np.diff(seq) < 0)[0]
            if non_monotonic_index.size > 0:
                if mode == 'filter':
                    seq = seq[:non_monotonic_index[0] + 1]
                elif mode == 'fix':
                    seq = fix_non_monotonic_sequence(seq)
            return seq

        return [filter_or_fix_sequence(seq, obs_size, mode) for seq in time]

    @staticmethod
    def worker(args):
        evaluator, chunk = args
        results = []
        for input_sample, type_sample, time_sample in chunk:
            recovered, recovered_input_sample, recovered_type_sample = evaluator.evaluate_sample(input_sample, type_sample)
            results.append((recovered, recovered_input_sample, recovered_type_sample, time_sample))
        return results

    def evaluate_samples(self, input_samples, type_samples, time_samples, use_multiprocessing, chunk_size=100):
        """Evaluate all samples and return results."""
        total_samples = len(input_samples)
        samples = list(zip(input_samples, type_samples, time_samples))
        
        # Split samples into chunks
        chunks = [samples[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
        args = [(self, chunk) for chunk in chunks]
        
        if use_multiprocessing:
            num_workers = self.config["num_workers"]
            with Pool(num_workers) as pool:
                chunk_results = list(tqdm(pool.imap(self.worker, args), total=len(chunks)))
        else:
            chunk_results = [self.worker(arg) for arg in tqdm(args, total=len(args))]
        
        # Flatten the results from all chunks
        results = [result for chunk in chunk_results for result in chunk]
        return self.aggregate_results(results)


    def evaluate_sample(self, input_sample, type_sample):
        """Evaluate a single sample for correctness and recovery."""
        # correct_events, incorrect_events = [], []
        correct_input_events, correct_type_events = [], []

        for i, (input_event, type_event) in enumerate(zip(input_sample, type_sample)):
            score, (recovered_input_event, recovered_type_event) = self.evaluate_event(input_event, type_event)
            if score:
                correct_input_events.append(recovered_input_event)
                correct_type_events.append(recovered_type_event)
        return len(correct_input_events) == len(input_sample), correct_input_events, correct_type_events

    def evaluate_event(self, input_event, type_event):
        """Check event syntax and semantics."""
        token_types = [token_type for token_type, _ in groupby(type_event)]
        decoded_event = TokenUtils.decode_tokens(self.tokenizer, input_event)
        table_name = decoded_event.split()[0]

        # Syntax validation
        if not self.validate_syntax(token_types, table_name):
            return 0, (input_event, type_event)

        # Attempt to recover the event
        recovered_input_event, recovered_type_event = self.recover_event(input_event, type_event, table_name)

        # If recovery fails, mark the event as incorrect
        if not recovered_input_event or not recovered_type_event:
            return 0, (input_event, type_event)

        return 1, (recovered_input_event, recovered_type_event)

    def validate_syntax(self, grouped_token_types, table_name):
        """Validate the syntax of the event."""
        token_type_counts = collections.Counter(grouped_token_types)
        
        table_token_type = self.p_token["table"]
        column_token_type = self.p_token["column"]
        content_token_type = self.p_token["content"]

        # 1. Check if all required token types are present
        if set(token_type_counts) != set(self.p_token.values()):
            return False # "[Syntax] Event does not consist of Table, Column, Content"
        
        # 2. Check if there is exactly one table token and it starts the event
        if token_type_counts[table_token_type] != 1 or grouped_token_types[0] != table_token_type:
            return False # "[Syntax] Table Token Type Error"

        # 3. Check if the number of column and content tokens match
        if token_type_counts[column_token_type] != token_type_counts[content_token_type]:
            return False # "[Syntax] Column/Content Matching Pair Error"

        # 4. Ensure the second token type (if present) is a column token
        if len(grouped_token_types) > 1 and grouped_token_types[1] != column_token_type:
            return False # "[Syntax] Column/Content Pair does not start with column Error"

        # 5. Verify the table name exists in the predefined vocabulary
        if table_name not in self.p_vocab:
            return False # "[Syntax] Table Name not in Predefined List"

        return True

    def recover_event(self, input_event, type_event, table_name):
        """Recover a potentially invalid event, processing grouped tokens."""
        recovered_input_event, recovered_type_event = [], []

        # Group tokens by their type
        for token_type, group in groupby(zip(input_event, type_event), key=lambda x: x[1]):
            tokens = [token for token, _ in group]  # Extract tokens for the current group
            decoded_tokens = TokenUtils.decode_tokens(self.tokenizer, tokens)
            
            if token_type == self.p_token["table"]:
                # Handle table tokens
                if decoded_tokens not in self.p_vocab:
                    return [], []  # Table name is invalid
                recovered_input_event.extend(tokens)
                recovered_type_event.extend([self.p_token["table"]] * len(tokens))

            elif token_type == self.p_token["column"]:
                if decoded_tokens not in self.p_vocab[table_name]:
                    # Handle column tokens
                    decoded_tokens = decoded_tokens.replace(' ', '') # Remove spaces
                    column_name = TokenUtils.find_closest_match(decoded_tokens, self.p_vocab.get(table_name, []))
                    encoded_column = TokenUtils.encode_text(self.tokenizer, column_name)
                    recovered_input_event.extend(encoded_column)
                    recovered_type_event.extend([self.p_token["column"]] * len(encoded_column))
                else:
                    column_name = decoded_tokens
                    recovered_input_event.extend(tokens)
                    recovered_type_event.extend([self.p_token["column"]] * len(tokens))

            elif token_type == self.p_token["content"]:
                # Handle content tokens
                recovered_content, is_valid = self.recover_content(tokens, table_name, column_name)
                if not is_valid:
                    return [], []  # Content recovery failed
                recovered_input_event.extend(recovered_content)
                recovered_type_event.extend([self.p_token["content"]] * len(recovered_content))

            else:
                raise ValueError(f"Unexpected token type encountered: {token_type}")

        return recovered_input_event, recovered_type_event
        
    def recover_content(self, content_token, table_name, column_name):
        """Recover a content token if the column is numerical."""
        # Check if the column is numerical
        numeric_col = self.p_vocab[table_name][column_name][1]  # Assume the second entry indicates numeric
        
        # Decode content_token (assumes content_token is already a list)
        decoded_content = TokenUtils.decode_tokens(self.tokenizer, content_token)
        decoded_content = "".join(decoded_content.split())
        if numeric_col:
            # Remove all spaces to ensure clean numeric validation
            decoded_content = "".join(decoded_content.split())

            # First check if the content is already a valid number
            if TokenUtils.is_number(decoded_content):
                return content_token, True  # Return the original token if it's already valid

            # If not, attempt to clean and validate the numeric value
            cleaned_value = TokenUtils.clean_numeric_content(decoded_content)
            if TokenUtils.is_number(cleaned_value):
                # Re-encode the valid numeric content
                dpe_value = ' '.join(cleaned_value)
                return TokenUtils.encode_text(self.tokenizer, dpe_value), True
            else:
                # Content cannot be recovered as a valid number
                return [], False
        else:
            # Non-numeric content is assumed valid
            return content_token, True

    def aggregate_results(self, results):
        """Aggregate evaluation results."""
        # Initialize lists for correct and incorrect sample indices
        correct_samples = []
        incorrect_samples = []
        correct_indices = []
        incorrect_indices = []

        # Iterate through the results to categorize them
        for idx, (recovered, input_sample, type_sample, time_sample) in enumerate(results):
            if recovered and len(time_sample) > 5:
                correct_samples.append((input_sample, type_sample, time_sample))
                correct_indices.append(idx)
            else:
                incorrect_samples.append((input_sample, type_sample, time_sample))
                incorrect_indices.append(idx)

        # Print summary of the results
        print(f"Correct samples: {len(correct_indices)}")
        print(f"Incorrect samples: {len(incorrect_indices)}")

        # Return the categorized results
        return correct_samples, incorrect_samples, correct_indices, incorrect_indices

    @staticmethod
    def parse_worker(args):
        """Worker function to process a chunk of samples."""
        evaluator, chunk = args
        processed_rows = []

        for stay_id, (input_sample, type_sample, time_sample) in chunk:

            res = list(zip(input_sample, type_sample))
            res = res[:len(time_sample)]

            for event_idx, (input_event, type_event) in enumerate(res):
                table_name = None
                row_data = {evaluator.pid_col: stay_id, evaluator.time_col: time_sample[event_idx] if event_idx < len(time_sample) else None}
                
                for token_type, group in groupby(zip(input_event, type_event), key=lambda x: x[1]):
                    tokens = [token for token, _ in group]
                    decoded_tokens = TokenUtils.decode_tokens(evaluator.tokenizer, tokens)

                    if token_type == evaluator.p_token["table"]:
                        table_name = decoded_tokens
                    elif token_type == evaluator.p_token["column"]:
                        # Better column name matching: try to find closest match in vocabulary
                        decoded_col = decoded_tokens.replace(' ', '')
                        # Try to match with predefined vocabulary
                        if table_name in evaluator.p_vocab:
                            try:
                                # Find closest match in vocabulary
                                current_column = TokenUtils.find_closest_match(
                                    decoded_col, 
                                    evaluator.p_vocab[table_name]
                                )
                            except:
                                current_column = decoded_col
                        else:
                            current_column = decoded_col
                    elif token_type == evaluator.p_token["content"]:
                        try:                            
                            if current_column in evaluator.col_type[table_name]["numeric_columns"]:
                                decoded_tokens = "".join(decoded_tokens.split())
                                value = float(decoded_tokens)
                                truncated_value = math.floor(value * 10000) / 10000
                                row_data[current_column] = truncated_value
                            elif current_column in evaluator.col_type[table_name]["categorical_columns"]: 
                                # Restore text and normalize format
                                restored = TokenUtils.restore_text(decoded_tokens)
                                
                                # Special handling for itemid columns: map text description to numeric ID
                                item_column = evaluator.config.get("item_column", {})
                                if table_name in item_column and current_column == item_column[table_name]:
                                    # This is an itemid column that needs mapping
                                    mapped_id = evaluator._map_itemid_text_to_id(table_name, restored)
                                    if mapped_id is not None:
                                        row_data[current_column] = mapped_id
                                    else:
                                        # If mapping fails, keep the text value (will be filtered in post-processing)
                                        row_data[current_column] = restored
                                else:
                                    # Regular categorical column
                                    row_data[current_column] = restored
                        except (ValueError, TypeError):
                            row_data[current_column] = None
                
                if table_name:
                    processed_rows.append((table_name, row_data))
        return processed_rows

    def normalize_dataframe_values(self, df, table_name):
        """
        Normalize categorical values in DataFrame to match real data format.
        This is a post-processing step to fix format mismatches.
        """
        if table_name not in self.col_type:
            return df
        
        categorical_columns = self.col_type[table_name].get("categorical_columns", [])
        
        # Load real data to get valid vocabulary (only for normalization)
        real_data_path = os.path.join(self.config["real_data_root"], f"{table_name}.csv")
        if os.path.exists(real_data_path):
            try:
                # Read only a sample to get vocabulary
                real_df_sample = pd.read_csv(real_data_path, nrows=10000)
                for col in categorical_columns:
                    if col in df.columns and col in real_df_sample.columns:
                        # Get valid vocabulary from real data
                        valid_vocab = set(real_df_sample[col].dropna().astype(str).unique())
                        
                        # Normalize values in generated data
                        def normalize_val(val):
                            if pd.isna(val):
                                return val
                            return TokenUtils.normalize_categorical_value(str(val), valid_vocab)
                        
                        df[col] = df[col].apply(normalize_val)
            except Exception as e:
                print(f"Warning: Could not normalize values for {table_name}: {e}")
        
        return df

    def parse_correct_samples_to_table(self, correct_samples, use_multiprocessing, chunk_size=100):
        """Parse correct samples into separate tables by table type, using chunk-based processing."""
        tables = collections.defaultdict(list)
        total_samples = len(correct_samples)

        # Use the provided num_workers or default to all available CPUs
        num_workers = self.config["num_workers"]

        # Split correct samples into chunks
        chunks = [
            [(i + stay_id, (input_sample, type_sample, time_sample))  # Global stay_id
            for stay_id, (input_sample, type_sample, time_sample) in enumerate(correct_samples[i:i + chunk_size])]
            for i in range(0, total_samples, chunk_size)
        ]

        args = [(self, chunk) for chunk in chunks]

        if use_multiprocessing:
            # Use multiprocessing to process chunks in parallel
            with Pool(num_workers) as pool:
                chunk_results = list(tqdm(pool.imap(self.parse_worker, args), total=len(chunks)))
        else:
            # Process chunks sequentially
            chunk_results = [self.parse_worker(arg) for arg in tqdm(args, total=len(chunks))]
            

        results = [result for chunk in chunk_results for result in chunk]
        for table_name, row_data in results:
            tables[table_name].append(row_data)

        table_dfs = {}
        for table_name, rows in tables.items():
            df = pd.DataFrame(rows)
            # Normalize categorical values to match real data format
            df = self.normalize_dataframe_values(df, table_name)
            table_dfs[table_name] = df
        
        print(f"Total samples: {total_samples}, Processed samples: {len(correct_samples)}")
        return table_dfs


def main(config):
    """Main execution pipeline."""
    evaluator = TableEvaluator(config)

    # Load and evaluate samples
    input_samples, type_samples, time_samples = evaluator.load_samples()
    correct_samples, _, _, incorrect_indices = evaluator.evaluate_samples(input_samples, type_samples, time_samples, config["use_multiprocessing"])
    np.save(os.path.join(config["syn_data_root"], "incorrect_indices.npy"), np.array(incorrect_indices))

    # Parse correct samples into a structured table
    parsed_tables = evaluator.parse_correct_samples_to_table(correct_samples, config["use_multiprocessing"])
    for table_name, df in parsed_tables.items():
        df.to_csv(f"{config['syn_data_root']}/{table_name}.csv", index=False)
    return parsed_tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ehr', type=str, required=True, choices=['mimiciv', 'eicu'], help='EHR dataset')
    parser.add_argument('--obs_size', type=int, default=12)
    parser.add_argument('--real_data_root', type=str, required=True)
    parser.add_argument('--syn_data_root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = get_config(
        ehr=args.ehr,
        obs_size=args.obs_size,
        real_data_root=args.real_data_root,
        syn_data_root=args.syn_data_root,
        seed=args.seed,
    )
    main(config)
    