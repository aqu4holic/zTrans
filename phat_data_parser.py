import pandas as pd
data = pd.read_parquet('/drive2/phatnt/zTrans/data/no_comment_dataset.parquet')

import numpy as np
seed = 18022004
np.random.seed(seed)

sample_cnt = 8000
data = data.sample(n = sample_cnt, random_state = seed)

import javalang

def remove_comments(source):
    source = source.replace('\r\n', '\n')
    state = "ETC"
    i = 0
    comments = []
    current_comment = None

    while i + 1 < len(source):
        if state == "ETC" and source[i] == '/' and source[i + 1] == '/':
            state = "LINE_COMMENT"
            current_comment = {
                "type": "LineComment",
                "range": {"start": i}
            }
            i += 2
            continue

        if state == "LINE_COMMENT" and source[i] == '\n':
            state = "ETC"
            current_comment["range"]["end"] = i
            comments.append(current_comment)
            current_comment = None
            i += 1
            continue

        if state == "ETC" and source[i] == '/' and source[i + 1] == '*':
            state = "BLOCK_COMMENT"
            current_comment = {
                "type": "BlockComment",
                "range": {"start": i}
            }
            i += 2
            continue

        if state == "BLOCK_COMMENT" and source[i] == '*' and source[i + 1] == '/':
            state = "ETC"
            current_comment["range"]["end"] = i + 2
            comments.append(current_comment)
            current_comment = None
            i += 2
            continue

        i += 1

    # Handle unfinished line comment at the end of the source code
    if current_comment and current_comment["type"] == "LineComment":
        if source[-1] == '\n':
            current_comment["range"]["end"] = len(source) - 1
        else:
            current_comment["range"]["end"] = len(source)
        comments.append(current_comment)

    # Remove the comments from the source code
    def remove_content(source, comments):
        result = []
        last_index = 0

        for comment in comments:
            start = comment["range"]["start"]
            end = comment["range"]["end"]
            # Append code before the comment
            result.append(source[last_index:start])
            # Update last index to skip the comment
            last_index = end

        # Append the remaining source code after the last comment
        result.append(source[last_index:])
        # Join and split lines, then filter out blank lines
        cleaned_code = ''.join(result).split('\n')
        return '\n'.join(line for line in cleaned_code if line.strip())

    return remove_content(source, comments)

def extract_methods_with_body(java_code):
    # print('java_code :',java_code)
    try:
        # Phân tích cú pháp mã Java
        tree = javalang.parse.parse(java_code)

        methods = []
        lines = java_code.splitlines()

        # Tìm các MethodDeclaration
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            # Lấy dòng bắt đầu của phương thức
            start_line = node.position.line - 1  # 0-based index

            # Xây dựng khai báo hàm đầy đủ (kể cả đa dòng)
            open_parens = 0
            method_declaration = []
            for line in lines[start_line:]:
                method_declaration.append(line)
                open_parens += line.count("(")
                open_parens -= line.count(")")
                if open_parens == 0 and ')' in line:  # Kết thúc khai báo hàm
                    break

            # Chuyển khai báo hàm thành chuỗi
            method_declaration_text = "\n".join(method_declaration).strip()

            # Lấy phần thân hàm (kể cả đa dòng)
            method_body = []
            open_braces = 0
            for line in lines[start_line:]:
                # print('line :',line)
                method_body.append(line)
                open_braces += line.count("{")
                open_braces -= line.count("}")
                if open_braces == 0 and len(method_body) > 1 and '}' in line:
                    break

            # Ghép lại toàn bộ hàm
            full_method_text =  "\n".join(method_body).strip()
            methods.append(full_method_text)

        return methods
    except:
        return None
# methods = extract_methods_with_body(java_code)

# # Hiển thị kết quả
# for method in methods:
#     print(method)
#     print("-" * 40)

def diff_methods(methods_start, methods_end):
    # Chuẩn hóa các phương thức để loại bỏ khoảng trắng không cần thiết
    normalized_start = [method.strip() for method in methods_start]
    normalized_end = [method.strip() for method in methods_end]

    # Tạo tập hợp để so sánh
    set_start = set(normalized_start)
    set_end = set(normalized_end)

    # Phương thức bị xóa hoặc thay đổi
    removed_methods = set_start - set_end

    # Phương thức mới hoặc thay đổi
    added_methods = set_end - set_start

    # Phương thức không thay đổi
    unchanged_methods = set_start & set_end

    return {
        "removed": removed_methods,
        "added": added_methods,
        "unchanged": unchanged_methods,
    }
# methods_start = extract_methods_with_body(data.iloc[1]['startCode'])
# methods_end = extract_methods_with_body(data.iloc[1]['endCode'])
def check_same_methods(method1, method2):
    name_method1 = method1.split('(')[0]
    name_method2 = method2.split('(')[0]
    return name_method1 == name_method2

# diff_result = diff_methods(methods_start, methods_end)
# row_changes = []
# removed = diff_result["removed"]
# added = diff_result["added"]

# remaining_added = list(added)

# for method1 in removed:
#     for method2 in list(remaining_added):  # Duyệt qua một bản sao của remaining_added
#         if check_same_methods(method1, method2):
#             row_changes.append([method1, method2])
#             remaining_added.remove(method2)  # Xóa method2 khỏi remaining_added
# for containmethods in remaining_added:
#     row_changes.append(['',containmethods])
# for i in row_changes:
#     for ex in i:
#         print(ex)
#         print('****')
#     print('----->')

# # Hiển thị kết quả
# print("Removed Methods:")
# for method in diff_result["removed"]:
#     print(method)
# print("-" * 40)

# print("Added Methods:")
# for method in diff_result["added"]:
#     print(method)
# print("-" * 40)


import pandas as pd

def process_and_expand_data(data):
    """
    Hàm xử lý và mở rộng DataFrame gốc bằng cách thêm các cột methods_before và methods_after.
    """
    expanded_data = []
    count = 0

    for index, row in data.iterrows():
        try:

            startCode = row['startCode']
            endCode = row['endCode']


            methods_start = extract_methods_with_body(startCode)
            methods_end = extract_methods_with_body(endCode)


            diff_result = diff_methods(methods_start, methods_end)
            removed = diff_result["removed"]
            added = diff_result["added"]

            row_changes = []
            remaining_added = list(added)

            for method1 in removed:
                have_func = False
                for method2 in list(remaining_added):
                    if check_same_methods(method1, method2):
                        row_changes.append([method1, method2])
                        remaining_added.remove(method2)
                        have_func = True
                        break
                if have_func == False:
                    row_changes.append([method1,''])

            for containmethods in remaining_added:
                row_changes.append(['',containmethods])


            for method1, method2 in row_changes:
                expanded_data.append({
                    **row.to_dict(),
                    'methods_before': method1,
                    'methods_after': method2
                })

        # for method1 in removed:
        #     if not any(method1 == change[0] for change in row_changes):
        #         expanded_data.append({
        #             **row.to_dict(),
        #             'methods_before': method1,
        #             'methods_after': ''
        #         })

        # for method2 in remaining_added:
        #     expanded_data.append({
        #         **row.to_dict(),
        #         'methods_before': '',
        #         'methods_after': method2
        #     })
        except Exception as e:
            # print(e)
            # import traceback
            # traceback.print_exc()
            print('Err at index:',count)
            print('--->')
        count+=1

    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df


expanded_data = process_and_expand_data(data)

expanded_data['methods_before'] = expanded_data['methods_before'].apply(lambda x: remove_comments(x))
expanded_data['methods_after'] = expanded_data['methods_after'].apply(lambda x: remove_comments(x))

# print(expanded_data)

# expanded_data = expanded_data.drop(['method_before','method_after'],axis=1)
expanded_data = expanded_data.drop(['startCode','endCode', 'repoSplitName', 'repoOwner', 'diff', 'total_added', 'total_removed', 'total_removed', 'total_position', 'detailed_changes', 'lib_percentage',
       'annotation_change'],axis=1)

expanded_data.to_parquet('/drive2/phatnt/zTrans/data/data_method.parquet')