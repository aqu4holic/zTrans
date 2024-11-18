import pandas as pd

from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import difflib

from tqdm import tqdm


# Initialize the parser and set the Java language
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)
def extract_methods_with_body(java_code):
    methods = []
    # print('java_code :',java_code)
    try:
        try:
            java_code = java_code.encode('utf-8')
        except Exception as e:
            print(e)
        tree = parser.parse(java_code)
        root_node = tree.root_node
        # Function to extract code from a node
        def extract_code(source_code, node):
            start_byte = node.start_byte
            end_byte = node.end_byte
            return source_code[start_byte:end_byte].decode("utf-8")

        # Traverse the syntax tree and find the method declaration
        for child in root_node.children:
            if child.type == "class_declaration":
                for class_child in child.children:
                    if class_child.type == "class_body":
                        for body_child in class_child.children:
                            if body_child.type == "method_declaration":
                                # Check for valid method declaration without errors
                                if not any(c.type == "ERROR" for c in body_child.children):
                                    method_name = ""
                                    method_signature = ""
                                    method_body = ""
                                    modifiers = []
                                    return_type = ""

                                    # Extract components of the method declaration
                                    for method_child in body_child.children:
                                        if method_child.type == "modifiers":
                                            modifiers = [extract_code(java_code, modifier) for modifier in method_child.children]
                                        elif method_child.type in ["type", "type_identifier", "scoped_type_identifier"]:  # Handle nested type nodes
                                            return_type = extract_code(java_code, method_child)
                                        elif method_child.type == "identifier":  # Capture method name
                                            method_name = extract_code(java_code, method_child)
                                        elif method_child.type == "formal_parameters":  # Capture parameters
                                            parameters = extract_code(java_code, method_child)
                                            method_signature = f"{' '.join(modifiers)} {return_type} {method_name}{parameters}"
                                    method_body = extract_code(java_code, body_child)
                                    methods.append({
                                        "name": method_name,
                                        "signature": method_signature.strip(),
                                        "body": method_body
                                    })


        return methods
    except Exception as e:
        print('Loi :' ,e)
        return None

# Function to remove comments
def remove_comments(java_code: str) -> str:
    # Parse the code
    tree = parser.parse(bytes(java_code, "utf8"))
    root_node = tree.root_node

    # Gather ranges of comment nodes
    comment_ranges = []
    def visit_node(node):
        if node.type in {"line_comment", "block_comment"}:
            comment_ranges.append((node.start_byte, node.end_byte))
        for child in node.children:
            visit_node(child)

    visit_node(root_node)

    # Remove comments by excluding their byte ranges
    result_code = bytearray(java_code, "utf8")
    for start, end in reversed(comment_ranges):  # Reverse to avoid shifting indices
        del result_code[start:end]

    return result_code.decode("utf8")

def diff_methods(methods_start, methods_end):
    """
    Compare methods based on their full dictionaries (e.g., name, signature, body).
    """
    # Normalize methods for comparison
    normalized_start = [{key: method[key].strip() if isinstance(method[key], str) else method[key] for key in method} for method in methods_start]
    normalized_end = [{key: method[key].strip() if isinstance(method[key], str) else method[key] for key in method} for method in methods_end]

    # Convert lists of methods to sets of frozensets for comparison
    set_start = set(frozenset(item.items()) for item in normalized_start)
    set_end = set(frozenset(item.items()) for item in normalized_end)

    # Determine differences
    removed_methods = [dict(items) for items in (set_start - set_end)]  # Methods in start but not in end
    added_methods = [dict(items) for items in (set_end - set_start)]    # Methods in end but not in start
    unchanged_methods = [dict(items) for items in (set_start & set_end)]  # Methods in both

    return {
        "removed": removed_methods,
        "added": added_methods,
        "unchanged": unchanged_methods,
    }

def check_same_methods(method1, method2):
    name_method1 = method1.split('(')[0]
    name_method2 = method2.split('(')[0]
    return name_method1 == name_method2

import pandas as pd

def get_diff(string1, string2):
    # Normalize by removing leading/trailing whitespace and replacing tabs with spaces
    normalized1 = [line.strip().replace('\t', '') for line in string1.splitlines()]
    normalized2 = [line.strip().replace('\t', '') for line in string2.splitlines()]

    # Generate the diff
    diff = difflib.unified_diff(
        normalized1,
        normalized2,
        lineterm=''
    )
    return '\n'.join(diff)

def process_and_expand_data(data):
    # t = ""
    """
    Hàm xử lý và mở rộng DataFrame gốc bằng cách thêm các cột methods_before và methods_after.
    """
    expanded_data = []
    count = 0
    total_methods_before = []
    total_methods_after = []
    for index in tqdm(range(len(data)), desc = 'parsing data'):
        try:
            row = data.iloc[index]

            startCode = row['startCode']
            endCode = row['endCode']
            startCode_clean = remove_comments(startCode)
            endCode_clean = remove_comments(endCode)

            data.at[index, 'startCode_cleaned'] = startCode_clean
            data.at[index, 'endCode_cleaned'] = endCode_clean

            methods_start = extract_methods_with_body(startCode_clean)
            methods_end = extract_methods_with_body(endCode_clean)

            total_methods_before = len(methods_start)
            total_methods_after = len(methods_end)

            diff_result = diff_methods(methods_start, methods_end)
            removed = diff_result["removed"]
            added = diff_result["added"]

            row_changes = []
            remaining_added = list(added)

            for method1 in removed:
                have_func = False
                for method2 in list(remaining_added):
                    if check_same_methods(method1['body'], method2['body']):
                        row_changes.append([method1['body'], method2['body'],method1['name'],method1['signature']])
                        remaining_added.remove(method2)
                        have_func = True
                        break
                if have_func == False:
                    row_changes.append([method1['body'],'',method1['name'],method1['signature']])

            for containmethods in remaining_added:
                row_changes.append(['',containmethods['body'],containmethods['name'],containmethods['signature']])

            for method1, method2,name,signature in row_changes:
                expanded_data.append({
                    **row.to_dict(),
                    "total_methods_before": total_methods_before,
                    "total_methods_after": total_methods_after,
                    'method_before': method1,
                    'method_after': method2,
                    'method_diff': get_diff(method1, method2),
                    'method_name' : name,
                    'method_signature' : signature,
                })

            # if (index % 1000 == 0):
            #     tmp = pd.DataFrame(expanded_data)
            #     tmp.to_parquet('/drive2/phatnt/zTrans/data/data_method_tresitter1.parquet')
            #     print('Saved at index:', index)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            print('Err at index:',count)
            print('--->')
        count += 1
    # data['total_methods_before'] = total_methods_before
    # data['total_methods_after'] = total_methods_after
    # data.to_parquet('/drive2/phatnt/zTrans/data/no_comment_dataset.parquet')
    expanded_df = pd.DataFrame(expanded_data)
    # with open(f'tmp.txt', 'w+') as file:
    #     file.write(t)

    return expanded_df

def main():
    print('Start to get data:')
    data = pd.read_parquet('no_comment_dataset.parquet')
    expanded_data = process_and_expand_data(data)
    expanded_data = expanded_data.drop(['startCode','endCode', 'repoSplitName', 'repoOwner', 'diff', 'total_added', 'total_removed', 'total_removed', 'total_position', 'detailed_changes', 'lib_percentage',
       'annotation_change', 'diff_cleaned'],axis=1)

    expanded_data.reset_index(drop = True)
    expanded_data['id'] = expanded_data.index

    expanded_data.to_parquet('data_method_treesitter1.parquet')
    print(len(expanded_data))
main()

# data = pd.read_parquet('/drive2/phatnt/zTrans/data/data_method.parquet')
# data = data.drop(['startCode','endCode', 'repoSplitName', 'repoOwner', 'diff', 'total_added', 'total_removed', 'total_removed', 'total_position', 'detailed_changes', 'lib_percentage',
#        'annotation_change'],axis=1)
# data.to_parquet('/drive2/phatnt/zTrans/data/data_method_treesitter.parquet')
# data = data[:100]
# data.to_parquet('/drive2/phatnt/zTrans/data/data_method_100.parquet')