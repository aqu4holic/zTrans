import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import difflib
from typing import Any, Dict, List
import time
from difflib import unified_diff
from tqdm import tqdm


# Initialize the parser and set the Java language
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

def extract_method_details(node, class_name, source_code):
    method_details = {
        'class_name': class_name,
        'body': '',
        'modifiers': '',
        'return_type': '',
        'name': '',
        'parameters': '',
        'signature': '',
    }

    start_byte, end_byte = node.start_byte, node.end_byte
    method_details['body'] = source_code[start_byte:end_byte].decode('utf-8', errors = 'replace')

    for child in node.children:
        if (child.type == 'modifiers'):
            method_details['modifiers'] = ' '.join([modifier.text.decode('utf-8') for modifier in child.children])
        elif ('type' in child.type):  # Return type
            method_details['return_type'] = child.text.decode('utf-8')
        elif (child.type == 'identifier'):  # Method name
            method_details['name'] = child.text.decode('utf-8')
        elif (child.type.endswith('parameters')):  # Parameter list
            param_string = ', '.join([param.text.decode('utf-8') for param in child.children if param.type.endswith('parameter')])
            method_details['parameters'] = param_string

    method_details['signature'] = f'{method_details["modifiers"]} {method_details["return_type"]} {method_details["name"]}({method_details["parameters"]})'
    method_details['signature_no_mod'] = f'{method_details["return_type"]} {method_details["name"]}({method_details["parameters"]})'

    return method_details

nested_class = []

def extract_methods_with_body(java_code, index):
    # print('java_code :',java_code)

    def has_errors(node):
        if node.type == 'ERROR':
            return True
        return any(has_errors(child) for child in node.children)

    def dfs_find_methods(index, node: Any, encoded_code, class_context: List[str] = None, ) -> List[Dict[str, Any]]:
        '''Perform DFS to find all method_declaration nodes and their enclosing class hierarchy.'''
        if class_context is None:
            class_context = []

        if (len(class_context) > 1):
            nested_class.append(index)

        method_details = []

        # If the node is a class, update the class context
        if node.type == 'class_declaration':
            class_name = None
            for child in node.children:
                if child.type == 'identifier':  # Class name
                    class_name = child.text.decode('utf-8')
                    break
            if class_name:
                class_context.append(class_name)

        # If the node is a method, extract its details with the full class hierarchy
        if node.type == 'method_declaration':
            full_class_name = '.'.join(class_context)  # Concatenate class names to show the hierarchy
            method_details.append(extract_method_details(node = node, class_name = full_class_name, source_code = encoded_code,))

        # Recursively process all children
        for child in node.children:
            method_details.extend(dfs_find_methods(index, child, encoded_code, class_context[:]))  # Pass a copy of class context

        # If the node is a class, pop the class name after processing its children
        if node.type == 'class_declaration' and class_context:
            class_context.pop()

        return method_details

    try:
        encoded_code = java_code.encode('utf-8')
        tree = parser.parse(encoded_code)
        root_node = tree.root_node

        if (has_errors(root_node)):
            raise Exception('Parsing errors found in the code')

        return dfs_find_methods(index, root_node, encoded_code)
    except Exception as e:
        print('Loi :' ,e)
        return None

# Function to remove comments
def remove_comments(java_code: str) -> str:
    # Parse the code
    tree = parser.parse(java_code.encode('utf-8'))
    root_node = tree.root_node

    # Gather ranges of comment nodes
    comment_ranges = []
    def visit_node(node):
        if node.type in {'line_comment', 'block_comment'}:
            comment_ranges.append((node.start_byte, node.end_byte, node.type))
        for child in node.children:
            visit_node(child)

    visit_node(root_node)

    # Remove comments by excluding their byte ranges
    result_code = bytearray(java_code, 'utf-8')
    for start, end, comment_type in reversed(comment_ranges):  # Reverse to avoid shifting indices
        if comment_type == 'block_comment':
            # Replace block comment with spaces
            result_code[start:end] = b' ' * (end - start + 1)
        else:
            # Remove line comments entirely
        #     # del result_code[start:end]
            result_code[start:end] = b' ' * (end - start + 1)
        # del result_code[start:end]

    return result_code.decode('utf-8').strip()

def diff_methods(methods_start, methods_end, lst_line_changed_in_files):
    '''
    Compare methods based on their full dictionaries (e.g., name, signature, body).
    '''
    # Normalize methods for comparison
    def normalize_methods(methods):
        res = []
        for method in methods:
            sub_method = {}

            sub_method['class_name'] = method['class_name'].strip()
            sub_method['name'] = method['name'].strip()
            sub_method['body'] = method['body'].strip()
            sub_method['modifiers'] = method['modifiers'].strip()
            sub_method['return_type'] = method['return_type'].strip()
            sub_method['parameters'] = method['parameters'].strip()
            sub_method['signature'] = method['signature'].strip()
            sub_method['signature_no_mod'] = method['signature_no_mod'].strip()

            res.append(sub_method)

        return res

    normalized_start = normalize_methods(methods_start)
    normalized_end = normalize_methods(methods_end)
    # normalized_start = [{key: method[key].strip() if isinstance(method[key], str) else method[key] for key in method} for method in methods_start]
    # normalized_end = [{key: method[key].strip() if isinstance(method[key], str) else method[key] for key in method} for method in methods_end]

    # Convert lists of methods to sets of frozensets for comparison
    set_start = set(frozenset(item.items()) for item in normalized_start)
    set_end = set(frozenset(item.items()) for item in normalized_end)

    # Determine differences
    removed_methods = [dict(items) for items in (set_start - set_end)]  # Methods in start but not in end
    added_methods = [dict(items) for items in (set_end - set_start)]    # Methods in end but not in start
    unchanged_methods = [dict(items) for items in (set_start & set_end)]  # Methods in both
    news_methods = []
    for method in removed_methods:
        methods_text_before = method['body'].replace(' ', '').replace('\n', '')
        for changed_line in lst_line_changed_in_files:
            check = True
            lst_line_before = changed_line['list_line_in_before']
            for line in lst_line_before:
                if line not in methods_text_before:
                    check = False
                    break
            if check:
                for method_ in added_methods:
                    check1 = True
                    method_text_after = method_['body'].replace(' ', '').replace('\n', '')
                    lst_line_after = changed_line['list_line_in_after']
                    for line in lst_line_after:
                        if line not in method_text_after:
                            check1 = False
                            break
                    if check1:
                        if method['name'].replace(' ','').replace('\n','') == method_['name'].replace(' ','').replace('\n',''):
                            news_methods.append({'method_before' : method,'method_after' : method_})
                            break
                
    
    for method in news_methods:
        if method['method_before'] in removed_methods:
            removed_methods.remove(method['method_before'])
        if method['method_after'] in added_methods:
            added_methods.remove(method['method_after'])
    return {
        'removed': removed_methods,
        'added': added_methods,
        'unchanged': unchanged_methods,
        'new_couple_methods' : news_methods
    }

def check_same_methods(method1, method2):
    '''
    Check if two methods are the same based on their full dictionaries.
    '''
    return (method1['signature'] == method2['signature']) and (method1['class_name'] == method2['class_name'])

def get_list_line_changed_in_files(startCode,endCode):
    lst_couple_line_method = []
    diff = get_diff_add_remove_using_difflib(startCode,endCode)
    # diff = diff_methods(lst_methods_start,lst_methods_end)
    list_diff = diff.splitlines()
    for index in range(len(list_diff)):
        if list_diff[index].startswith('@@'):
            lst_line_before = []
            lst_line_after = []
            index+=1
            s_before = ''
            s_after = ''
            while True:
                if  index >= len(list_diff) or list_diff[index].startswith('@@')  :
                    if s_before!='' and s_before!=' ':
                        lst_line_before.append(s_before)
                    if s_after!='' and s_after!=' ':
                        lst_line_after.append(s_after)
                    lst_couple_line_method.append({
                        'list_line_in_before' : lst_line_before,
                        'list_line_in_after' : lst_line_after
                    })
                    break
                
                if list_diff[index].startswith('-') and 'import' not in list_diff[index]:
                    if s_after != '':
                        lst_line_after.append(s_after)
                        s_after = ''
                    s_before += list_diff[index].replace('-',' ')
                elif list_diff[index].startswith('+') and 'import' not in list_diff[index]:
                    if s_before != '':
                        lst_line_before.append(s_before)
                        s_before = ''
                    s_after+= list_diff[index].replace('+',' ') 
                index+=1
    return lst_couple_line_method

def get_diff_add_remove_using_difflib(startCode,endCode):
    startCode = startCode.splitlines()
    endCode = endCode.splitlines()
    diff = unified_diff(
    startCode,
    endCode,
    lineterm='' 
    )
    result = ''
    for line in diff:
        result+=line
        result+='\n'
    return result

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
    # t = ''
    '''
    Hàm xử lý và mở rộng DataFrame gốc bằng cách thêm các cột methods_before và methods_after.
    '''
    expanded_data = []
    total_methods_before = []
    total_methods_after = []
    errors = []
    for index in tqdm(range(len(data)), desc = 'parsing data'):
        try:
            row = data.iloc[index].copy()

            startCode = row['startCode']
            endCode = row['endCode']

            # with open('before.txt', 'w') as file:
            #     file.write(startCode)
            # with open('after.txt', 'w') as file:
            #     file.write(endCode)

            startCode_cleaned = remove_comments(startCode)
            endCode_cleaned = remove_comments(endCode)

            row['startCode_cleaned'] = startCode_cleaned
            row['endCode_cleaned'] = endCode_cleaned

            # with open('before_cleaned.txt', 'w') as file:
            #     file.write(startCode_cleaned)
            # with open('after_cleaned.txt', 'w') as file:
            #     file.write(endCode_cleaned)

            methods_start = extract_methods_with_body(startCode_cleaned, index)
            methods_end = extract_methods_with_body(endCode_cleaned, index)
            # s_start = '\n'.join(line_start for line_start in methods_start)
            # s_end = '\n'.join(line_end for line_end in methods_end)
            # with open('list_methods_start.txt', 'w') as file:
            #     file.write(s_start)
            # with open('list_methods_end.txt', 'w') as file:
            #     file.write(s_end)
            total_methods_before = len(methods_start)
            total_methods_after = len(methods_end)
            lst_line_changed_in_files = get_list_line_changed_in_files(startCode_cleaned,endCode_cleaned)
            diff_result = diff_methods(methods_start, methods_end, lst_line_changed_in_files)
            removed = diff_result['removed']
            added = diff_result['added']
            new_couple_methods = diff_result['new_couple_methods']
            row_changes = []
            remaining_added = list(added)
            len_removed = len(removed)
            s = f'First length removed :{str(len_removed)}'
            # hanlde removed and added methods
            for method1 in removed:
                have_func = False
                for method2 in list(remaining_added):
                    if check_same_methods(method1, method2):
                        row_changes.append(
                            [
                                method1['class_name'],
                                method2['class_name'],

                                method1['name'],
                                method2['name'],

                                method1['body'],
                                method2['body'],

                                method1['modifiers'],
                                method2['modifiers'],

                                method1['return_type'],
                                method2['return_type'],

                                method1['parameters'],
                                method2['parameters'],

                                method1['signature'],
                                method2['signature'],

                                method1['signature_no_mod'],
                                method2['signature_no_mod'],
                            ]
                        )
                        remaining_added.remove(method2)
                        removed.remove(method1)
                        s+='\n'
                        s+= f'After length removed :{str(len(removed))}'
                        s+='\n'
                        have_func = True
                        break
                # if have_func == False:
                #     row_changes.append(
                #         [
                #             method1['class_name'],
                #             '',

                #             method1['name'],
                #             '',

                #             method1['body'],
                #             '',

                #             method1['modifiers'],
                #             '',

                #             method1['return_type'],
                #             '',

                #             method1['parameters'],
                #             '',

                #             method1['signature'],
                #             '',

                #             method1['signature_no_mod'],
                #             '',
                #         ]
                #     )
            # hanlde new_couple_methods
            for method in new_couple_methods:
                row_changes.append(
                            [
                                method['method_before']['class_name'],
                                method['method_after']['class_name'],

                                method['method_before']['name'],
                                method['method_after']['name'],

                                method['method_before']['body'],
                                method['method_after']['body'],

                                method['method_before']['modifiers'],
                                method['method_after']['modifiers'],

                                method['method_before']['return_type'],
                                method['method_after']['return_type'],

                                method['method_before']['parameters'],
                                method['method_after']['parameters'],

                                method['method_before']['signature'],
                                method['method_after']['signature'],

                                method['method_before']['signature_no_mod'],
                                method['method_after']['signature_no_mod'],
                            ]
                        )
            s += f'Final length removed :{str(len(removed))}'
            with open(f'log_method.txt', 'w') as file:
                file.write(s)
            for method1 in removed:
                row_changes.append(
                        [
                            method1['class_name'],
                            '',

                            method1['name'],
                            '',

                            method1['body'],
                            '',

                            method1['modifiers'],
                            '',

                            method1['return_type'],
                            '',

                            method1['parameters'],
                            '',

                            method1['signature'],
                            '',

                            method1['signature_no_mod'],
                            '',
                        ]
                    )
            
            for containmethods in remaining_added:
                row_changes.append(
                    [
                        '',
                        containmethods['class_name'],

                        '',
                        containmethods['name'],

                        '',
                        containmethods['body'],

                        '',
                        containmethods['modifiers'],

                        '',
                        containmethods['return_type'],

                        '',
                        containmethods['parameters'],

                        '',
                        containmethods['signature'],

                        '',
                        containmethods['signature_no_mod'],
                    ]
                )
            
            for (
                    method1_class, method2_class,
                    method1_name, method2_name,
                    method1, method2,
                    method1_modifiers, method2_modifiers,
                    method1_return_type, method2_return_type,
                    method1_param, method2_param,
                    method1_signature, method2_signature,
                    method1_signature_no_mod, method2_signature_no_mod,
                ) in row_changes:
                expanded_data.append({
                    **row.to_dict(),
                    'total_methods_before': total_methods_before,
                    'total_methods_after': total_methods_after,

                    'class_before': method1_class,
                    'class_after': method2_class,

                    'method_before_name' : method1_name,
                    'method_after_name' : method2_name,

                    'method_before': method1,
                    'method_after': method2,

                    'method_diff': get_diff(method1, method2),

                    'method_before_modifiers' : method1_modifiers,
                    'method_after_modifiers' : method2_modifiers,

                    'method_before_return_type' : method1_return_type,
                    'method_after_return_type' : method2_return_type,

                    'method_before_parameters' : method1_param,
                    'method_after_parameters' : method2_param,

                    'method_before_signature' : method1_signature,
                    'method_after_signature' : method2_signature,

                    'method_before_signature_no_mod' : method1_signature_no_mod,
                    'method_after_signature_no_mod' : method2_signature_no_mod,
                })

            # if (index % 1000 == 0):
            #     tmp = pd.DataFrame(expanded_data)
            #     tmp.to_parquet('/drive2/phatnt/zTrans/data/data_method_tresitter1.parquet')
            #     print('Saved at index:', index)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            print('Err at index:', index)
            print('--->')
            errors.append(index)

            # break

            with open(f'data/errors_method.txt', 'w') as file:
                for error in errors:
                    file.write(f'{error}\n')# break
    # data['total_methods_before'] = total_methods_before
    # data['total_methods_after'] = total_methods_after
    # data.to_parquet('/drive2/phatnt/zTrans/data/no_comment_dataset.parquet')
    expanded_df = pd.DataFrame(expanded_data)

    with open(f'data/errors_method.txt', 'w') as file:
        for error in errors:
            file.write(f'{error}\n')

    with open(f'data/nested_class.txt', 'w') as file:
        for nested in nested_class:
            file.write(f'{nested}\n')

    return expanded_df

def main():
    print('Start to get data:')
    data_prefix = 'data'
    # data = pd.read_parquet('no_comment_dataset.parquet')
    data = pd.read_parquet(f'{data_prefix}/migration_others_class_code_no_import.parquet')
    # data = data[data['fileName'] == 'utilities/src/main/java/ec/tstoolkit/utilities/Files2.java']
    expanded_data = process_and_expand_data(data)
    dropped_columns = ['startCode', 'endCode',
                       'repoSplitName', 'repoOwner', 'diff', 'total_added', 'total_removed', 'total_removed', 'total_position',
                       'detailed_changes', 'lib_percentage', 'diff_cleaned',
    ]
    # expanded_data = expanded_data.drop(['startCode','endCode', 'repoSplitName', 'repoOwner', 'diff',
    #                                     'total_added', 'total_removed', 'total_removed', 'total_position',
    #                                     'detailed_changes', 'lib_percentage',
    #                                     'diff_cleaned'], axis = 1)

    dropped_columns = set(dropped_columns) & set(expanded_data.columns)
    expanded_data = expanded_data.drop(dropped_columns, axis = 1)

    start_time = time.time()
    expanded_data.to_parquet(f'data/migration_others_method_update_by_quoc.parquet', engine = 'pyarrow')
    end_time = time.time()
    print(f'Finished in {(end_time - start_time):.2f} seconds')

    expanded_data.reset_index(drop = True)
    expanded_data['id'] = expanded_data.index

    dropped_columns = ['startCode_cleaned', 'endCode_cleaned',]
    dropped_columns = set(dropped_columns) & set(expanded_data.columns)

    expanded_data = expanded_data.drop(dropped_columns, axis = 1)

    start_time = time.time()
    expanded_data.to_parquet(f'data/migration_others_method_no_code_update_by_quoc.parquet', engine = 'pyarrow')
    end_time = time.time()
    print(f'Finished in {(end_time - start_time):.2f} seconds')

    print(len(expanded_data))

main()

# data = pd.read_parquet('/drive2/phatnt/zTrans/data/data_method.parquet')
# data = data.drop(['startCode','endCode', 'repoSplitName', 'repoOwner', 'diff', 'total_added', 'total_removed', 'total_removed', 'total_position', 'detailed_changes', 'lib_percentage',
#        'annotation_change'],axis=1)
# data.to_parquet('/drive2/phatnt/zTrans/data/data_method_treesitter.parquet')
# data = data[:100]
# data.to_parquet('/drive2/phatnt/zTrans/data/data_method_100.parquet')