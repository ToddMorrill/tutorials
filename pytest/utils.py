import string

def format_data_for_display(people):
    formatted_data = []
    for temp_dict in people:
        running_string = []
        running_string.append(temp_dict['given_name'])
        running_string.append(temp_dict['family_name'] + ':')
        running_string.append(temp_dict['title'])
        final_string = ' '.join(running_string)
        formatted_data.append(final_string)
    return formatted_data


def format_data_for_excel(people):
    cols = 'given,family,title'
    data_rows = []
    for temp_dict in people:
        running_string = []
        running_string.append(temp_dict['given_name'])
        running_string.append(temp_dict['family_name'])
        running_string.append(temp_dict['title'])
        final_string = ','.join(running_string)
        data_rows.append(final_string)
    formatted_data = '\n'.join([cols] + data_rows)
    return formatted_data

def is_palindrome(input_string):
    input_string = input_string.replace(' ', '').lower()
    input_string = input_string.translate(str.maketrans('', '', string.punctuation))
    return input_string == input_string[::-1]