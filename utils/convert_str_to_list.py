def convert_str_to_list(text:str):
    """convert a text to list of int

    Args:
        text (str): the text to convert
    """
    if ',' in text:
        converted_text = [int(i) for i in text.split(',')]
    else:
        converted_text = int(text)
    return converted_text